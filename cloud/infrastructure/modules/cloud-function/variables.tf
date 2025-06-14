variable "region" {}
variable "project" {}
variable "function_name" {}
variable "function_entry_point" {}
variable "env" {
  type        = map(string)
  description = "(Optional) Environment variables."
  default     = {}
}
variable "event_trigger" {
  type = list(object({
    event_type = string
    resource   = string
    failure_policy = object({
      retry = bool
    })
  }))
  description = "(Optional) A source that fires events in response to a condition in another service. Structure is documented below. Cannot be used with `trigger_http`."
  default     = []
}
