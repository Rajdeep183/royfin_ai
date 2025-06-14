"use client"

import * as React from "react"
import { Check, Search } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { searchTickers } from "@/lib/stock-api"

interface ComboboxProps {
  options?: { value: string; label: string }[]
  onSelect: (value: string) => void
  placeholder?: string
  isLoading?: boolean
  noOptionsMessage?: string
}

export function Combobox({ 
  options = [], 
  onSelect, 
  placeholder = "Select an option...",
  isLoading = false,
  noOptionsMessage = "No options found."
}: ComboboxProps) {
  const [open, setOpen] = React.useState(false)
  const [value, setValue] = React.useState("")
  const [searchQuery, setSearchQuery] = React.useState("")
  const [searchResults, setSearchResults] = React.useState<{ value: string; label: string }[]>([])
  const [searching, setSearching] = React.useState(false)

  // Use either provided options or search results
  const displayOptions = searchQuery ? searchResults : options

  // Search function with debouncing
  const performSearch = React.useCallback(async (query: string) => {
    if (!query.trim()) {
      setSearchResults([])
      return
    }

    setSearching(true)
    try {
      const results = await searchTickers(query)
      setSearchResults(results)
    } catch (error) {
      console.error("Search error:", error)
      setSearchResults([])
    } finally {
      setSearching(false)
    }
  }, [])

  // Debounced search effect
  React.useEffect(() => {
    const timer = setTimeout(() => {
      performSearch(searchQuery)
    }, 300)

    return () => clearTimeout(timer)
  }, [searchQuery, performSearch])

  const handleSelect = (currentValue: string) => {
    setValue(currentValue)
    setOpen(false)
    setSearchQuery("")
    onSelect(currentValue)
  }

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value)
  }

  const selectedLabel = displayOptions.find((option) => option.value === value)?.label || value

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button 
          variant="outline" 
          role="combobox" 
          aria-expanded={open} 
          className="w-full justify-between"
          disabled={isLoading}
        >
          {value ? selectedLabel : placeholder}
          <Search className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-full p-0" align="start">
        <div className="border-b p-3">
          <Input
            placeholder="Search stocks..."
            value={searchQuery}
            onChange={handleSearchChange}
            className="h-9"
            autoFocus
          />
        </div>
        <div className="max-h-60 w-full overflow-auto">
          {searching && (
            <div className="py-2 px-3 text-sm text-muted-foreground text-center">
              Searching...
            </div>
          )}
          {!searching && displayOptions.length === 0 && (
            <div className="py-2 px-3 text-sm text-muted-foreground text-center">
              {searchQuery ? "No stocks found. Try a different search." : noOptionsMessage}
            </div>
          )}
          {!searching && displayOptions.map((option) => (
            <div
              key={option.value}
              onClick={() => handleSelect(option.value)}
              className={cn(
                "relative cursor-pointer select-none py-2 pl-8 pr-4 text-sm hover:bg-accent hover:text-accent-foreground",
                value === option.value ? "bg-accent text-accent-foreground" : "text-foreground"
              )}
            >
              <span className="absolute left-2 top-1/2 -translate-y-1/2">
                <Check className={cn("h-4 w-4", value === option.value ? "opacity-100" : "opacity-0")} />
              </span>
              <div className="truncate">{option.label}</div>
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  )
}
