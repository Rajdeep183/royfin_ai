# configure terraform to use google cloud as provider

provider "google" {
  project = var.tf_project
  region  = var.tf_region
}
