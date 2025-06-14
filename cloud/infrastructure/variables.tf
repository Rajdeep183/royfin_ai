variable "tf_project" {
  description = "The GCP project ID"
  type        = string
}

variable "tf_region" {
  description = "The GCP region"
  type        = string
  default     = "us-west1"
}
variable "tf_version" {
  type    = string
  default = "0.0.0-SNAPSHOT"
}

locals {
  environment_variables = { for tuple in regexall("(.*)=(.*)", file("../.env")) : tuple[0] => sensitive(tuple[1]) }
}
