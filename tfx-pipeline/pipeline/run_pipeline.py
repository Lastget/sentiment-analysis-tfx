# definition of the TFX pipeline, definition of all components and parameters
from tfx.proto import example_gen_pb2, trainer_pb2
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.dsl.components.common.resolver import Resolver
from typing import List, Optional, Text, Dict
from tfx.v1.dsl import Importer
from tfx.types import standard_artifacts
from tfx import v1 as tfx
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
import tensorflow_model_analysis as tfma
from tfx.orchestration import pipeline
from ml_metadata.proto import metadata_store_pb2

def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    preprocessing_module: Text,
    tuner_path: Text,
    training_module: Text,
    serving_model_dir: Text,
    beam_pipeline_args: Optional[List[Text]] = None,
    vertex_training_args: Optional[Dict[Text, Text]] = None,
    vertex_serving_args: Optional[Dict[Text, Text]] = None,
    enable_cache: Optional[bool] = False,
    use_gpu: bool = False,
    region: Optional[Text] = 'us-central1',
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    ) -> tfx.dsl.Pipeline:
    """Implements the pipeline with TFX."""
    
    
    # initialize components list
    components = []

    # Brings data into the pipeline or otherwise joins/converts training data.
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(
                name='train', hash_buckets=3),
            example_gen_pb2.SplitConfig.Split(
                name='eval', hash_buckets=1)
        ]))

    example_gen = CsvExampleGen(
        input_base=data_path, output_config=output)
        
    components.append(example_gen)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples'])
        
    components.append(statistics_gen)   


    # Generates schema based on statistics files.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
        
    components.append(schema_gen)

    # Performs anomaly detection, missing,  based on statistics and data schema.
    example_validator = ExampleValidator(  # pylint: disable=unused-variable
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
        
    components.append(example_validator)

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=preprocessing_module,
    )

    components.append(transform)

    # Import Hyperparameters  
    hparams_importer = Importer(
    source_uri=tuner_path,
    # This part assigns an ID ('import_hparams') to the imported hyperparameters. 
    artifact_type=standard_artifacts.HyperParameters).with_id('import_hparams')
    
    components.append(hparams_importer)


    # Configuration for Vertex AI Training.
    if use_gpu:
        # See https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec#acceleratortype
        # for available machine types.
        vertex_training_args['worker_pool_specs'][0]['machine_spec'].update({
            'accelerator_type': 'NVIDIA_TESLA_K80',
            'accelerator_count': 1
        })

    # Train 
    trainer_args = dict(
        module_file=training_module,
        examples=transform.outputs['transformed_examples'],
        hyperparameters=hparams_importer.outputs['result'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=tfx.proto.TrainArgs(splits=['train']),
        eval_args=tfx.proto.EvalArgs(splits=['eval']),
    )

    if vertex_training_args is not None:
        trainer_args['custom_config'] = {
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
              True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
              region,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
              vertex_training_args,
            'use_gpu':
              use_gpu,
        }   

        trainer = tfx.extensions.google_cloud_ai_platform.Trainer(**trainer_args)
    else:
        trainer = Trainer(**trainer_args)
        
    components.append(trainer)

    #  Resolver: specify the last base model within the Evaluator component.
    model_resolver = Resolver(
    strategy_class=LatestBlessedModelStrategy,
    model=Channel(type=Model),
    model_blessing=Channel(
        type=ModelBlessing)
        )
        
    components.append(model_resolver)

    # ensuring they are good enough to be sent to production
    # Uses TFMA to compute a evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    eval_config = tfma.EvalConfig(
    # This specifies the model(s) to be evaluated. 
    model_specs=[tfma.ModelSpec(label_key='sentiment')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[

            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='FalsePositives'),
            tfma.MetricConfig(class_name='TruePositives'),
            tfma.MetricConfig(class_name='FalseNegatives'),
            tfma.MetricConfig(class_name='TrueNegatives'),
            tfma.MetricConfig(
                class_name='SparseCategoricalAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0}),
                    # Change threshold will be ignored if there is no
                    # baseline model resolved from MLMD (first run).
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': -1e-10}
                    )
                )
            )
        ])
    ])

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)
        
    components.append(evaluator)

    # Pusher
    # pre-build image from vertex 
    # See https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
    # for available container images.
    serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'

    if use_gpu:
        vertex_serving_args.update({
            'accelerator_type': 'NVIDIA_TESLA_K80',
            'accelerator_count': 1
        })

        serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest'
    
    pusher_args = {
    'model': trainer.outputs['model'],
    'model_blessing': evaluator.outputs['blessing'],
    }


    if vertex_serving_args is not None:
        pusher_args['custom_config'] = {
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
                True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
                region,
            tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY:
                serving_image,
            tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY:
                vertex_serving_args,
        }

        pusher = tfx.extensions.google_cloud_ai_platform.Pusher(**pusher_args)
    else:
        pusher_args['push_destination'] = tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory=serving_model_dir))
        # Pushes the model to a file destination if check passed.
        pusher = Pusher(**pusher_args)
        
    components.append(pusher)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        beam_pipeline_args=beam_pipeline_args,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
    )


    

