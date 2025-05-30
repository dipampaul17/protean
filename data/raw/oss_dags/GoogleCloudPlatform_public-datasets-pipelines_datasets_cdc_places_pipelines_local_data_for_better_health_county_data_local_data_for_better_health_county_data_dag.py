# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from airflow import DAG
from airflow.providers.cncf.kubernetes.operators import kubernetes_pod
from airflow.providers.google.cloud.transfers import gcs_to_bigquery

default_args = {
    "owner": "Google",
    "depends_on_past": False,
    "start_date": "2021-03-01",
}


with DAG(
    dag_id="cdc_places.local_data_for_better_health_county_data",
    default_args=default_args,
    max_active_runs=1,
    schedule_interval="@daily",
    catchup=False,
    default_view="graph",
) as dag:

    # Run CSV transform within kubernetes pod
    local_data_transform_csv = kubernetes_pod.KubernetesPodOperator(
        task_id="local_data_transform_csv",
        startup_timeout_seconds=600,
        name="cdc_places_local_data_for_better_health_county_data",
        namespace="composer-user-workloads",
        service_account_name="default",
        config_file="/home/airflow/composer_kube_config",
        image_pull_policy="Always",
        image="{{ var.json.cdc_places.container_registry.run_csv_transform_kub }}",
        env_vars={
            "SOURCE_URL": "https://chronicdata.cdc.gov/resource/swc5-untb.csv",
            "SOURCE_FILE": "files/data.csv",
            "TARGET_FILE": "files/data_output.csv",
            "TARGET_GCS_BUCKET": "{{ var.value.composer_bucket }}",
            "TARGET_GCS_PATH": "data/cdc_places/local_data_for_better_health_county_data/data_output.csv",
            "CSV_HEADERS": '["year","stateabbr","statedesc","locationname","datasource","category","measure","data_value_unit","data_value_type","data_value","data_value_footnote_symbol","data_value_footnote","low_confidence_limit","high_confidence_limit","totalpopulation","locationid","categoryid","measureid","datavaluetypeid","short_question_text","geolocation"]',
            "RENAME_MAPPINGS": '{"year": "year","stateabbr": "stateabbr","statedesc": "statedesc","locationname": "locationname","datasource": "datasource","category": "category","measure": "measure","data_value_unit": "data_value_unit","data_value_type": "data_value_type","data_value": "data_value","data_value_footnote_symbol": "data_value_footnote_symbol","data_value_footnote": "data_value_footnote","low_confidence_limit": "low_confidence_limit","high_confidence_limit": "high_confidence_limit","totalpopulation": "totalpopulation","locationid": "locationid","categoryid": "categoryid","measureid": "measureid","datavaluetypeid": "datavaluetypeid","short_question_text": "short_question_text","geolocation": "geolocation"}',
            "PIPELINE_NAME": "local_data_for_better_health_county_data",
        },
        container_resources={
            "memory": {"request": "32Gi"},
            "cpu": {"request": "2"},
            "ephemeral-storage": {"request": "10Gi"},
        },
    )

    # Task to load CSV data to a BigQuery table
    load_local_data_to_bq = gcs_to_bigquery.GCSToBigQueryOperator(
        task_id="load_local_data_to_bq",
        bucket="{{ var.value.composer_bucket }}",
        source_objects=[
            "data/cdc_places/local_data_for_better_health_county_data/data_output.csv"
        ],
        source_format="CSV",
        destination_project_dataset_table="cdc_places.local_data_for_better_health_county_data",
        skip_leading_rows=1,
        write_disposition="WRITE_TRUNCATE",
        schema_fields=[
            {"name": "year", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "stateabbr", "type": "STRING", "mode": "NULLABLE"},
            {"name": "statedesc", "type": "STRING", "mode": "NULLABLE"},
            {"name": "locationname", "type": "STRING", "mode": "NULLABLE"},
            {"name": "datasource", "type": "STRING", "mode": "NULLABLE"},
            {"name": "category", "type": "STRING", "mode": "NULLABLE"},
            {"name": "measure", "type": "STRING", "mode": "NULLABLE"},
            {"name": "data_value_unit", "type": "STRING", "mode": "NULLABLE"},
            {"name": "data_value_type", "type": "STRING", "mode": "NULLABLE"},
            {"name": "data_value", "type": "FLOAT", "mode": "NULLABLE"},
            {
                "name": "data_value_footnote_symbol",
                "type": "STRING",
                "mode": "NULLABLE",
            },
            {"name": "data_value_footnote", "type": "STRING", "mode": "NULLABLE"},
            {"name": "low_confidence_limit", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "high_confidence_limit", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "totalpopulation", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "locationid", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "categoryid", "type": "STRING", "mode": "NULLABLE"},
            {"name": "measureid", "type": "STRING", "mode": "NULLABLE"},
            {"name": "datavaluetypeid", "type": "STRING", "mode": "NULLABLE"},
            {"name": "short_question_text", "type": "STRING", "mode": "NULLABLE"},
            {"name": "geolocation", "type": "GEOGRAPHY", "mode": "NULLABLE"},
        ],
    )

    local_data_transform_csv >> load_local_data_to_bq
