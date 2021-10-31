# ST558 Project 2
## Sergio Mora and Ashley Ko

Below are the reports for each category. Also is the code that creates said reports automatically based on the `params` parameter. Each report is simmilar to the other reports but can vary depending on the data.

  - [lifestyle](lifestyle.html)
  - [entertainment](entertainment.html)
  - [bus](bus.html)
  - [socmed](socmed.html)
  - [tech](tech.html)
  - [world](world.html)


This is the code that creates the above reports.

```markdown
library(rmarkdown)
# Creates a list of the six data_channel_is_* options
category <- c("lifestyle", "entertainment", "bus", "socmed", "tech", "world")

# Creates output filenames
output_file <- paste0(category, ".html")
# Creates a list for each data channel with just the category parameter
params = lapply(category, FUN = function(x){list(category = x)})

# Stores list of file output file names and parameters as a data frame
reports <- tibble(output_file, params)

# Applies the render function to each pair of output file name and category
apply(reports, MARGIN = 1, FUN = function(x){ render(input = "Project2.Rmd",
                                                     output_file = x[[1]],
                                                     params = x[[2]])})
```
