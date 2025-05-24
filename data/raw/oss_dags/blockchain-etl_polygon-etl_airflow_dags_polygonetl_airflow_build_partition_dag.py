from __future__ import print_function

import logging
from datetime import datetime, timedelta

from airflow import models
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from google.cloud import bigquery

from utils.error_handling import handle_dag_failure

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def build_partition_dag(
        dag_id,
        partitioned_project_id,
        partitioned_dataset_name ,
        public_project_id,
        public_dataset_name,
        load_dag_id,
        partition_start_date=datetime(2015, 7, 30),
        partition_schedule='0 0 * * *',
        notification_emails=None,
):

    default_dag_args = {
        'depends_on_past': False,
        'start_date': partition_start_date,
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 5,
        'retry_delay': timedelta(minutes=5),
        'on_failure_callback': handle_dag_failure,
    }

    if notification_emails and len(notification_emails) > 0:
        default_dag_args['email'] = [email.strip() for email in notification_emails.split(',')]

    # Define a DAG (directed acyclic graph) of tasks.
    dag = models.DAG(
        dag_id,
        catchup=False,
        schedule=partition_schedule,
        default_args=default_dag_args)

    def add_partition_tasks(task, sql_template, dependencies=None):
        def enrich_task(ds, **kwargs):
            client = bigquery.Client()

            ds_with_underscores = ds.replace('-', '_')
            sql = sql_template.format(
                ds=ds,
                ds_with_underscores=ds_with_underscores,
                partitioned_project_id=partitioned_project_id,
                partitioned_dataset_name=partitioned_dataset_name,
                public_project_id=public_project_id,
                public_dataset_name=public_dataset_name,
            )
            print(sql)
            query_job = client.query(sql)
            result = query_job.result()
            logging.info(result)

        extract_operator = PythonOperator(
            task_id=f'partition_{task}',
            python_callable=enrich_task,
            execution_timeout=timedelta(minutes=60),
            dag=dag
        )

        if dependencies is not None and len(dependencies) > 0:
            for dependency in dependencies:
                dependency >> extract_operator
        return extract_operator

    wait_for_ethereum_load_dag_task = ExternalTaskSensor(
        task_id='wait_for_polygon_load_dag',
        external_dag_id=load_dag_id,
        external_task_id='load_metadata_task',
        execution_delta=timedelta(hours=1),
        priority_weight=0,
        mode='reschedule',
        poke_interval=5 * 60,
        timeout=60 * 60 * 12,
        dag=dag)

    partition_logs_task = add_partition_tasks('logs', SQL_TEMPLATE_LOGS)
    partition_traces_task = add_partition_tasks('traces', SQL_TEMPLATE_TRACES)

    wait_for_ethereum_load_dag_task >> partition_logs_task
    wait_for_ethereum_load_dag_task >> partition_traces_task

    done_task = BashOperator(
        task_id='done',
        bash_command='echo done',
        dag=dag
    )

    partition_logs_task >> done_task
    partition_traces_task >> done_task

    return dag


SQL_TEMPLATE_LOGS = '''
CREATE OR REPLACE TABLE
  `{partitioned_project_id}.{partitioned_dataset_name}.logs_by_date_{ds_with_underscores}`
PARTITION BY
  RANGE_BUCKET(_topic_partition_index, GENERATE_ARRAY(0, 3999, 1))
OPTIONS(
  expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 72 HOUR)
)
AS

SELECT 
  MOD(ABS(FARM_FINGERPRINT(topics[SAFE_OFFSET(0)])), 3999) as _topic_partition_index, 
  *
FROM `{public_project_id}.{public_dataset_name}.logs`
WHERE date(block_timestamp) = '{ds}'
'''

SQL_TEMPLATE_TRACES = '''
CREATE OR REPLACE TABLE
  `{partitioned_project_id}.{partitioned_dataset_name}.traces_by_date_{ds_with_underscores}`
PARTITION BY
  RANGE_BUCKET(_input_partition_index, GENERATE_ARRAY(0, 3999, 1))
OPTIONS(
  expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 72 HOUR)
)
AS

SELECT 
  MOD(ABS(FARM_FINGERPRINT(SUBSTRING(input, 0, 10))), 3999) as _input_partition_index, 
  *
FROM `{public_project_id}.{public_dataset_name}.traces`
WHERE date(block_timestamp) = '{ds}'
'''
