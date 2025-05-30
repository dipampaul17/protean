from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.state import DagRunState
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import timedelta, datetime
from betflow.historical.config import ProcessingConfig
from dotenv import load_dotenv, find_dotenv
from airflow.models import Variable
from betflow.historical.hist_utils import (
    check_source_data,
    upload_glue_script,
    create_or_update_glue_job,
)

load_dotenv(find_dotenv("my.env"), override=True)

default_args = {
    "owner": ProcessingConfig.OWNER,
    "depends_on_past": True,
    "email_on_failure": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "tags": ["processing", "historical", "odds"],
}

with DAG(
    "odds_batch_processing",
    default_args=default_args,
    description="Process all sports odds data into Iceberg tables",
    # schedule_interval="@daily",
    # schedule_interval="5 0 * * *",
    schedule_interval=None,  # triggered by parent DAG
    catchup=True,
    # start_date=datetime(2022, 7, 31),
    start_date=datetime(2024, 12, 9),
    # end_date=datetime(2024, 12, 1),
    max_active_runs=16,  # Increase concurrent DAG runs
    concurrency=32,  # Increase task concurrency
) as dag:
    wait_for_odds = ExternalTaskSensor(
        task_id="wait_for_odds",
        external_dag_id="odds_ingestion_dqc",
        external_task_id=None,  # Wait for entire DAG
        allowed_states=[DagRunState.SUCCESS],
        failed_states=[DagRunState.FAILED],
        execution_date_fn=lambda dt: dt,
        mode="poke",
        timeout=7200,  # Increase timeout to prevent premature failures
        poke_interval=60,
        soft_fail=True,
        check_existence=True,
    )
    with TaskGroup("common_tasks") as common_tasks:
        upload_script = PythonOperator(
            task_id="upload_odds_script",
            python_callable=upload_glue_script,
            op_kwargs={"args": {"type": "odds"}},
            provide_context=True,
        )

    for sport, config in ProcessingConfig.SPORT_CONFIGS.items():
        with TaskGroup(group_id=f"{sport}_tasks") as sport_tasks:
            check_data = ShortCircuitOperator(
                task_id=f"check_{sport}_data",
                python_callable=check_source_data,
                op_kwargs={"args": {"type": "odds", "sport": sport}},
                provide_context=True,
            )

            setup_glue_job = PythonOperator(
                task_id=f"setup_{sport}_odds_job",
                python_callable=create_or_update_glue_job,
                op_kwargs={"args": {"type": "odds", "sport": sport}},
                provide_context=True,
            )

            process_odds = GlueJobOperator(
                task_id=f"process_{sport}_odds",
                job_name=f"{sport}_odds_processing",
                s3_bucket=Variable.get("LOGS_BUCKET"),
                script_args={
                    "--JOB_NAME": f"{sport}_odds_processing",
                    "--date": "{{ ds }}",  # macros.ds_add(ds, -1)
                    "--source_path": f"s3://{ProcessingConfig.S3_PATHS['raw_bucket']}/{ProcessingConfig.S3_PATHS['odds_prefix']}/{sport}/",
                    "--database_name": ProcessingConfig.GLUE_DB["db_name"],
                    "--table_name": ProcessingConfig.GLUE_DB[f"{sport}_odds_table"],
                    "--warehouse_path": f"s3://{ProcessingConfig.S3_PATHS['processing_bucket']}/processed",
                },
                region_name="us-east-1",
                trigger_rule="all_done",
            )

            check_data >> setup_glue_job >> process_odds

        wait_for_odds >> common_tasks
        common_tasks >> [sport_tasks for sport in ProcessingConfig.SPORT_CONFIGS]
