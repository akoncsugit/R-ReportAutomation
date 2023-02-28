# Predicting Number of Shares from Online News Popularity Data with Automated Reports
### Ashley Ko and Sergio Mora 

This repo is used to generate prediction of the number of shares based on data from [online news popularity data set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) for each of the six `data_channel_is_*` (`lifestyle`, `entertainment`, `bus`, `socmed`, `tech`, and `world`). The libraries required for this project are `knitr`, `rmarkdown`, `caret`, `tidyvers`, `corrplot`, and `tree`.


Below are the reports for each category of **data_channel_is_***. Also is the code that creates said reports automatically based on the `params` parameter. Each report is similar to the other reports but can vary depending on the data.

  - [Lifestyle](lifestyle.html)
  - [Entertainment](entertainment.html)
  - [Business](bus.html)
  - [Social Media](socmed.html)
  - [Technology](tech.html)
  - [World](world.html)


This is the code that creates the above reports.

```markdown
library(rmarkdown)
# Creates a list of the six data_channel_is_* options
category <- c("lifestyle", "entertainment", "bus", "socmed", "tech", "world")

# Creates output filenames
output_file <- paste0(category, ".md")
# Creates a list for each data channel with just the category parameter
params = lapply(category, FUN = function(x){list(category = x)})

# Stores list of file output file names and parameters as a data frame
reports <- tibble(output_file, params)

# Applies the render function to each pair of output file name and category
apply(reports, MARGIN = 1, FUN = function(x){ render(input = "Project2.Rmd",
                                                     output_file = x[[1]],
                                                     params = x[[2]])})
```
