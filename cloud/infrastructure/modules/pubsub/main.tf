resource "google_pubsub_topic" "topic" {
  name    = "${var.tf_stage}_${var.name}"
  project = var.project
}
