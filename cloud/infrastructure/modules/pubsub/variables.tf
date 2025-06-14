variable "name" {
    type = string
    description = "Pub/Sub name"
}

variable "tf_stage" {
    type = string
    description = "stage name"
}

variable "tf_version" {
    type = string
    description = "Pub/Sub deployment version"
    default = "0.0.0-SNAPSHOT"
}

variable "project" {
    type = string
    description = "Project to create Pub/Sub in"
}
