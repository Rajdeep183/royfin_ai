locals {
  tf_stage = terraform.workspace
}

// Pubsub
module "cron_to_sprint_checkpointer_pubsub" {
  source = "./modules/pubsub"

  name = "cron-to-sprint-checkpointer-pubsub"

  project    = var.tf_project
  tf_stage   = local.tf_stage
  tf_version = var.tf_version
}

// Daily Cloud Scheduler Job
resource "google_cloud_scheduler_job" "daily_scheduler" {
  name        = "daily-function-scheduler"
  description = "Triggers Cloud Functions daily"
  schedule    = "0 0 * * *" # Daily at midnight
  time_zone   = "UTC"
  region      = "us-central1" # Currently cloud scheduler not available in us-west1

  pubsub_target {
    topic_name = module.cron_to_sprint_checkpointer_pubsub.google_pubsub_topic_id
    data       = base64encode("{\"message\": \"Trigger Cloud Function\"}")
  }
}

// Cloud Function for sprint_checkpointer, runs based on cloud scheduler
module "sprint_checkpointer" {
  source  = "./modules/cloud-function"
  project = var.tf_project

  function_name        = "sprint_checkpointer"
  function_entry_point = "main"

  region = var.tf_region

  env = local.environment_variables

  event_trigger = [{
    event_type = "google.pubsub.topic.publish"
    resource   = module.cron_to_sprint_checkpointer_pubsub.google_pubsub_topic_id
    failure_policy = {
      retry = false
    }
  }]

}
