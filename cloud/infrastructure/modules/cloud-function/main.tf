locals {
  timestamp = formatdate("YYMMDDhhmmss", timestamp()) # Generate a timestamp for the zip file name
}

data "archive_file" "source" {
  type        = "zip"
  source_dir  = "${path.root}/../src" # Directory where your Python source code is
  output_path = "${path.root}/src-${var.function_name}-${local.timestamp}.zip"
}

resource "google_storage_bucket" "bucket" {
  name     = "${var.function_name}-cf"
  location = "US"
  project  = var.project
}

resource "google_storage_bucket_object" "zip" {
  depends_on = [google_storage_bucket.bucket]
  name       = "${var.function_name}_${data.archive_file.source.output_md5}.zip"
  bucket     = google_storage_bucket.bucket.name
  source     = data.archive_file.source.output_path
}

resource "google_cloudfunctions_function" "function" {
  depends_on = [google_storage_bucket.bucket]
  name       = var.function_name
  runtime    = "python312"
  timeout    = 180
  region     = var.region

  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.bucket.name
  source_archive_object = google_storage_bucket_object.zip.name
  entry_point           = var.function_entry_point

  max_instances = 5

  environment_variables = var.env
  dynamic "event_trigger" {
    for_each = var.event_trigger == null ? [] : var.event_trigger
    content {
      event_type = event_trigger.value["event_type"]
      resource   = event_trigger.value["resource"]
      dynamic "failure_policy" {
        for_each = event_trigger.value["failure_policy"] == null ? [] : [1]
        content {
          retry = event_trigger.value["failure_policy"].retry
        }
      }
    }
  }

}

output "trigger_url" {
  description = "URL used to trigger cloud function."
  value       = google_cloudfunctions_function.function.https_trigger_url
}

