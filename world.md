---
title: "ST558 Project 2"
author: "Sergio Mora & Ashley Ko"
output:
  html_document:  
    df_print: paged 
    keep_md: yes
params:
  category: "lifestyle"
#params:
#  category: category
---



# Introduction
This project summarizes and create and compares predictions for online news popularity data sets about articles from Mashable. The goal is to predict the number of `shares` for each of the six data channels (lifestyle, entertainment, business, social media, tech, world). A description and summary of this data set, `OnlineNewsPopularity.csv` can be found here [Online News Popularity Data Set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).

In order to achieve this goal, we first subset data based on predictors we hypothesize to have an impact the number of shares of per article. We choose following predictors `n_tokens_title`, `n_tokens_content`, `num_imgs`, `num_videos`, `average_token_length`, `kw_avg_avg`, `is_weekend`, `global_subjectivity`, `global_sentiment_polarity`, `global_rate_negative_words`, `avg_negative_polarity`, `abs_title_subjectivity`, `abs_title_sentiment_polarity`.

First, we selected average key words(kw_avg_avg), average length of words (average_token_length), content (n_tokens_content), and title (n_tokens_title) as metrics for the readability of an article. We believe that easy to read articles might be more likely to be shared. In tandem with readability, articles with more images(num_imgs) and videos(num_videos) might be easier to consume and more likely shared. It is possible that the number of shares is linked whether an article is published on a weekday or weekend (is_weekend). Generally speaking, people have more more free time on the weekends which leads to more screen-time and perhaps, more articles being shared. Lastly, emotionally charged content is often compelling so we investigated the impact of the subjectivity(global_subjective and abs_title_subjectivity), sentiment polarity(global_sentiment_polarity and abs_title_sentiment_polarity), and negative word rate (global_rate_negative_words).

A variety of numerical and graphic summaries have been used to display our chosen predictors and their relationship to the number of shares for the training data set. We generated the following numeric summaries minimums, 1st quartiles, medians, means, 3rd quartiles, maximums, standard deviations, IQRs, and correlations. Additionally, we used correlation plots, box plots, rug plots, and a few others to visualize the training. We used both linear models and ensemble methods (random forest and boosted tree) to generate predictions.

## Libraries and Set-Up
This report uses the a `set.seed(123)` and the following libraries `knitr`, `rmarkdown`, `caret`, `tidyvers`, `corrplot`, `tree`, and `parallel`.

# Data

## Reading in data
We will read in our csv dataset. As instructed we will also split our data by `data_channel_is_*`.

```r
#Reads in csv file `OnlineNewsPopularity/OnlineNewsPopularity.csv` to produce a data frame
online_new_popularity_data <- read.csv("./OnlineNewsPopularity/OnlineNewsPopularity.csv")
```

## Subsetting the data
We will subset the data based on the category listed in our YAML header. In this case, using data from `data_channel_is_world`. We will remove non-predictors such as `url` and `timedelta` and selected our desired predictors** and `shares`.
**`n_tokens_title`, `n_tokens_content`, `num_imgs`, `num_videos`, `average_token_length`, `kw_avg_avg`, `is_weekend`, `global_subjectivity`, `global_sentiment_polarity`, `global_rate_negative_words`, `avg_negative_polarity`, `abs_title_subjectivity`, `abs_title_sentiment_polarity`

```r
# Subsetting our data based on the category parameter, dropping non-predictors,
# and subsetting for desired predictors and response variables
subset_data <- online_new_popularity_data %>%
  filter(!!as.name(paste0("data_channel_is_",params$category)) == 1) %>%
  select(n_tokens_title, n_tokens_content, num_imgs:average_token_length,
         kw_avg_avg, is_weekend, global_subjectivity, global_sentiment_polarity, 
         global_rate_negative_words, avg_negative_polarity, abs_title_subjectivity,
         abs_title_sentiment_polarity, shares)
```

Next, we will check for potential problematic values such as NA or infinity. These could result in errors with later analysis. Should a problem arise later on, this allows for a diagnostic to rule out potential problematic values.

```r
# Checking data for NA  or infinite values
na_or_infinite <- as.data.frame(apply(subset_data, 2, function(x) any(is.na(x) | is.infinite(x))))
colnames(na_or_infinite) <- c("NA or Infinite values")
na_or_infinite %>% kable()
```



|                             |NA or Infinite values |
|:----------------------------|:---------------------|
|n_tokens_title               |FALSE                 |
|n_tokens_content             |FALSE                 |
|num_imgs                     |FALSE                 |
|num_videos                   |FALSE                 |
|average_token_length         |FALSE                 |
|kw_avg_avg                   |FALSE                 |
|is_weekend                   |FALSE                 |
|global_subjectivity          |FALSE                 |
|global_sentiment_polarity    |FALSE                 |
|global_rate_negative_words   |FALSE                 |
|avg_negative_polarity        |FALSE                 |
|abs_title_subjectivity       |FALSE                 |
|abs_title_sentiment_polarity |FALSE                 |
|shares                       |FALSE                 |

In this chunk, we will split our newly subset data frame into training and test data sets.
We will use a simple 70/30 split

```r
# Setting up a simple 70/30 split for our already subset data
sample_size <- floor(0.7 * nrow(subset_data))
train_ind <- sample(seq_len(nrow(subset_data)), size = sample_size)

# This will be needed later on when we start modeling
training_data <- subset_data[train_ind,]
test_data <- subset_data[-train_ind,]
```

# Summarizations

## Numeric Summary

### Six Number Summary
First, let's perform a simple six number summary of all vvariables from the training data set. This summary includes minimum, 1st quartile, median, mean, 3rd quartile, and maximum values. This provides a senses of scale and range for variable values. Note: Binary response variable`is_weekend` has as values of 0 or 1. 

```r
#Generates a six number summary from training_data
summary <- summary(training_data)
summary
```

```
##  n_tokens_title  n_tokens_content    num_imgs         num_videos     
##  Min.   : 4.00   Min.   :   0.0   Min.   :  0.000   Min.   : 0.0000  
##  1st Qu.: 9.00   1st Qu.: 335.0   1st Qu.:  1.000   1st Qu.: 0.0000  
##  Median :10.00   Median : 512.0   Median :  1.000   Median : 0.0000  
##  Mean   :10.57   Mean   : 599.8   Mean   :  2.839   Mean   : 0.5427  
##  3rd Qu.:12.00   3rd Qu.: 770.0   3rd Qu.:  2.000   3rd Qu.: 1.0000  
##  Max.   :20.00   Max.   :7081.0   Max.   :100.000   Max.   :51.0000  
##  average_token_length   kw_avg_avg      is_weekend     global_subjectivity
##  Min.   :0.000        Min.   :    0   Min.   :0.0000   Min.   :0.0000     
##  1st Qu.:4.656        1st Qu.: 2062   1st Qu.:0.0000   1st Qu.:0.3575     
##  Median :4.821        Median : 2394   Median :0.0000   Median :0.4148     
##  Mean   :4.677        Mean   : 2516   Mean   :0.1311   Mean   :0.4028     
##  3rd Qu.:4.974        3rd Qu.: 2773   3rd Qu.:0.0000   3rd Qu.:0.4659     
##  Max.   :5.783        Max.   :17839   Max.   :1.0000   Max.   :0.8658     
##  global_sentiment_polarity global_rate_negative_words
##  Min.   :-0.35947          Min.   :0.00000           
##  1st Qu.: 0.02272          1st Qu.:0.01105           
##  Median : 0.07236          Median :0.01656           
##  Mean   : 0.07610          Mean   :0.01704           
##  3rd Qu.: 0.12498          3rd Qu.:0.02215           
##  Max.   : 0.50000          Max.   :0.07504           
##  avg_negative_polarity abs_title_subjectivity abs_title_sentiment_polarity
##  Min.   :-0.9000       Min.   :0.0000         Min.   :0.0000              
##  1st Qu.:-0.3083       1st Qu.:0.2000         1st Qu.:0.0000              
##  Median :-0.2458       Median :0.5000         Median :0.0000              
##  Mean   :-0.2512       Mean   :0.3623         Mean   :0.1275              
##  3rd Qu.:-0.1917       3rd Qu.:0.5000         3rd Qu.:0.2000              
##  Max.   : 0.0000       Max.   :0.5000         Max.   :1.0000              
##      shares       
##  Min.   :   35.0  
##  1st Qu.:  833.2  
##  Median : 1100.0  
##  Mean   : 2230.4  
##  3rd Qu.: 1800.0  
##  Max.   :96500.0
```

### Standard Deviation
The previous section does not generate standard deviation for the variable values. Standard deviation is necessary for determining the variance of the response and predictors. It is a good diagnostic to spot potential issues that violate assumptions necessary for models and analysis. Here we will produce standard deviations for each variable from `training_data.` Note: Binary response variable`is_weekend` has as values of 0 or 1. 

```r
# Removes scientific notation for readability
options(scipen = 999)

# Generates and rounds (for simplicity) standard deviation
train_SDs <- as_tibble(lapply(training_data[, 1:14], sd))
round(train_SDs, digits = 5)
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"2.07491","2":"419.943","3":"5.34447","4":"1.62061","5":"0.86891","6":"835.4018","7":"0.3375","8":"0.10732","9":"0.08072","10":"0.00917","11":"0.10977","12":"0.18021","13":"0.20356","14":"4692.952"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

```r
# Turns scientific notation on
options(scipen = 0)
```

### IQR
Although the 1st and 3rd quartiles are identified in the six number summary, it is helpful quantify the range between these two values or IQR. IQR is also needed for subsequent plotting. Note: Binary response variable`is_weekend` has as values of 0 or 1. 

```r
# Uses lapply to generate IQR for all variables in training_data
IQR<- as_tibble(lapply(training_data[, 1:14], IQR))
IQR
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"3","2":"435","3":"1","4":"1","5":"0.3182165","6":"710.991","7":"0","8":"0.1084153","9":"0.102251","10":"0.01110217","11":"0.1166667","12":"0.3","13":"0.2","14":"966.75"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

### Correlations
Prior to preforming any model fitting or statistical analysis, it is essential to understand the potential correlation among predictors and between the response and predictors. Correlation helps identify potential collinearity and, thus, allows for better candidate model selection. It is worth noting any absolute correlation values > 0.5. However, this threshold has been left to discretion of the individual.

```r
# Creating a data frame of a single column of variable names 
variables <- as_tibble(attributes(training_data)$names) %>%
  rename(variable = "value")

# Generates correlations for all variables in training_data
corr <- cor(training_data)
correlations <- as_tibble(round(corr, 3))

# Binds the variable names to the correlation data frame
corr_mat <- bind_cols(variables, correlations)
correlation_matrix <- column_to_rownames(corr_mat, var = "variable")

correlation_matrix 
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"1.000","2":"0.063","3":"-0.007","4":"0.027","5":"-0.036","6":"0.000","7":"0.044","8":"-0.031","9":"-0.061","10":"0.011","11":"-0.031","12":"-0.134","13":"0.044","14":"0.048","_rn_":"n_tokens_title"},{"1":"0.063","2":"1.000","3":"0.241","4":"0.060","5":"0.235","6":"-0.009","7":"0.037","8":"0.185","9":"0.011","10":"0.155","11":"-0.156","12":"-0.018","13":"-0.001","14":"-0.020","_rn_":"n_tokens_content"},{"1":"-0.007","2":"0.241","3":"1.000","4":"-0.031","5":"-0.225","6":"0.106","7":"0.009","8":"-0.186","9":"-0.016","10":"-0.138","11":"0.089","12":"-0.011","13":"0.048","14":"0.084","_rn_":"num_imgs"},{"1":"0.027","2":"0.060","3":"-0.031","4":"1.000","5":"0.002","6":"0.072","7":"0.012","8":"0.056","9":"0.025","10":"0.006","11":"-0.052","12":"-0.023","13":"0.047","14":"0.056","_rn_":"num_videos"},{"1":"-0.036","2":"0.235","3":"-0.225","4":"0.002","5":"1.000","6":"-0.078","7":"-0.003","8":"0.614","9":"0.127","10":"0.313","11":"-0.387","12":"0.016","13":"-0.023","14":"-0.085","_rn_":"average_token_length"},{"1":"0.000","2":"-0.009","3":"0.106","4":"0.072","5":"-0.078","6":"1.000","7":"0.013","8":"0.009","9":"0.051","10":"-0.022","11":"0.027","12":"-0.025","13":"0.055","14":"0.102","_rn_":"kw_avg_avg"},{"1":"0.044","2":"0.037","3":"0.009","4":"0.012","5":"-0.003","6":"0.013","7":"1.000","8":"-0.039","9":"-0.025","10":"-0.016","11":"-0.004","12":"0.010","13":"0.003","14":"0.041","_rn_":"is_weekend"},{"1":"-0.031","2":"0.185","3":"-0.186","4":"0.056","5":"0.614","6":"0.009","7":"-0.039","8":"1.000","9":"0.339","10":"0.254","11":"-0.446","12":"-0.042","13":"0.091","14":"0.021","_rn_":"global_subjectivity"},{"1":"-0.061","2":"0.011","3":"-0.016","4":"0.025","5":"0.127","6":"0.051","7":"-0.025","8":"0.339","9":"1.000","10":"-0.447","11":"0.230","12":"-0.044","13":"0.060","14":"0.030","_rn_":"global_sentiment_polarity"},{"1":"0.011","2":"0.155","3":"-0.138","4":"0.006","5":"0.313","6":"-0.022","7":"-0.016","8":"0.254","9":"-0.447","10":"1.000","11":"-0.242","12":"-0.057","13":"0.042","14":"-0.023","_rn_":"global_rate_negative_words"},{"1":"-0.031","2":"-0.156","3":"0.089","4":"-0.052","5":"-0.387","6":"0.027","7":"-0.004","8":"-0.446","9":"0.230","10":"-0.242","11":"1.000","12":"-0.028","13":"-0.057","14":"0.006","_rn_":"avg_negative_polarity"},{"1":"-0.134","2":"-0.018","3":"-0.011","4":"-0.023","5":"0.016","6":"-0.025","7":"0.010","8":"-0.042","9":"-0.044","10":"-0.057","11":"-0.028","12":"1.000","13":"-0.413","14":"-0.005","_rn_":"abs_title_subjectivity"},{"1":"0.044","2":"-0.001","3":"0.048","4":"0.047","5":"-0.023","6":"0.055","7":"0.003","8":"0.091","9":"0.060","10":"0.042","11":"-0.057","12":"-0.413","13":"1.000","14":"0.033","_rn_":"abs_title_sentiment_polarity"},{"1":"0.048","2":"-0.020","3":"0.084","4":"0.056","5":"-0.085","6":"0.102","7":"0.041","8":"0.021","9":"0.030","10":"-0.023","11":"0.006","12":"-0.005","13":"0.033","14":"1.000","_rn_":"shares"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

## Graphical summaries
We suppose that part of what makes a link shareable is how easy it is for the content to be consumed. Smaller, bite sized information, in a simple writing style might be easier to process and result in more shares.
We will test this out via proxy's. We will measure shares against average key words(kw_avg_avg), average length of words (average_token_length), average number of words in the content (n_tokens_content), and number of words in the title (n_tokens_title). The idea here is to measure both the quantity of words as well as the complexity of the content. i.e. an article with 500 "easy" words could be shared more than an article with 100 "difficult" words.

### Improving graphical summaries - Cutlier check 
To produce graphs and plots that give an accurate sense of the data, we will search for potential. If we have any outliers we will remove them first to get an idea of what the bulk of shares come from. We will follow what the boxplot tells us when choosing what to remove.

```r
# Boxplot from training_data
boxplot(training_data$shares,horizontal = TRUE, range = 2, main = "Boxplot of shares with outliers")
```

![](world_files/figure-html/boxplot-outliers-1.png)<!-- -->

```r
boxplot(training_data$shares,horizontal = TRUE, range = 2, outline = FALSE,main = "Boxplot of shares without outliers")
```

![](world_files/figure-html/boxplot-outliers-2.png)<!-- -->

```r
# Creates a subset of training data without outliers for #plotting purposes
IQR <- quantile(training_data$shares)[4] - quantile(subset_data$shares)[2]
upper_limit <- quantile(training_data$shares)[4] + (1.5 * IQR)
lower_limit <- quantile(training_data$shares)[2] - (1.5 * IQR)
subset_data_wo_outliers <- training_data %>% filter(shares <= upper_limit & shares >= lower_limit)
```

After we remove any potential outliers to our data our we can compare shares our key metrics.

```r
correlation1 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$kw_avg_avg)

plot1 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = kw_avg_avg)) + 
  geom_point() +
  geom_smooth() +
  labs(title = "Number of shares vs. Average number of key words", y= "# of shares", x = "Average # of key words") +
  geom_text(color = "red",x=15000,y=5000,label = paste0("Correlation = ",round(correlation1,3)))

plot1
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
```

![](world_files/figure-html/shares-vs-keywords-average-1.png)<!-- -->

We can measure the trend of shares as a function of Average number of key words. If we see a possitive trend we can say that the more key words in the articles the more likely it is to be shared, the opposite can also be said. We measure the correlation to get a more precise gauge in case the graph is not clear enough.

```r
correlation2 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$average_token_length)

plot2 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = average_token_length)) +
geom_density_2d() + 
  labs(title = "number of shares vs. Average length of words in content", y= "# of shares", x = "Average length of words in content") +
  geom_text(color = "red",x=5,y=3500,label = paste0("Correlation = ",round(correlation2,3)))

plot2
```

![](world_files/figure-html/shares-vs-average-length-of-words-in-content-1.png)<!-- -->

With a density plot as a function of average length of words in content we see where most of our shares come from. We can utilize this to help explain our model down below.


```r
correlation3 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$n_tokens_content)

plot3 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = n_tokens_content)) +
geom_rug() +
  labs(title = "number of shares vs. number of words in content", y= "# of shares", x = "# of words in content") +
  geom_text(color = "red",x=4000,y=4000,label = paste0("Correlation = ",round(correlation3,3)))

plot3
```

![](world_files/figure-html/density-plot-1.png)<!-- -->

Using a rug graph we can measure the relationship between number of words in content and the number of shares. The intersection between where both rugs are highly concentrated is where how we can measure correlation. If both rugs are concentrated near zero than we see that the less words the more shareable the articles are or vice versa.


```r
correlation4 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$n_tokens_title)

plot4 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = n_tokens_title)) +
geom_col() +
  labs(title = "number of shares vs. number of words in title", y= "# of shares", x = "# of words in title") +
  geom_text(color = "red",x=15,y=600000,label = paste0("Correlation = ",round(correlation4,3)))

plot4
```

![](world_files/figure-html/rug-graph-1.png)<!-- -->
We see how the `# of words in title` as distributed with respect to number of shares. Any large skewness would be a flag for us to research further.


Here we graphically depict the correlations among the variables in the training data.

```r
# Note need to probably name graphs still etc...work in progress
corrplot(cor(training_data), tl.col = "black")
```

![](world_files/figure-html/correlation-plot-1.png)<!-- -->

In addition to the previous correlation plot, it is necessary to look at individual scatterplots of the shares vs. predictors. Previously, we plotted the relationship between shares and various word counts, but in this section we will focus exclusively on the emotion predictors such as `global_subjectivity`, `global_sentiment_polarity`,
`global_rate_negative_words`, `avg_negative_polarity`, `abs_title_subjectivity`, and `abs_title_sentiment_polarity`.

```r
corNoOut<- function(x) {
  var1 <- get(x, subset_data_wo_outliers)
  title <-paste0("Scatterplot of shares vs ", x)
  xlabel <- x
  g <- ggplot(subset_data_wo_outliers, aes(x = var1, y = shares)) + geom_point() +
    geom_smooth() + geom_smooth(method = lm, col = "Red") +
    labs(title = title, y= "# of shares", x = xlabel)
}
sentiment_preds <- list("global_subjectivity", "global_sentiment_polarity",
"global_rate_negative_words", "avg_negative_polarity",
"abs_title_subjectivity", "abs_title_sentiment_polarity")
lapply(sentiment_preds, corNoOut)
```

```
## [[1]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
```

```
## `geom_smooth()` using formula 'y ~ x'
```

![](world_files/figure-html/sentiment-correlation-1.png)<!-- -->

```
## 
## [[2]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](world_files/figure-html/sentiment-correlation-2.png)<!-- -->

```
## 
## [[3]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](world_files/figure-html/sentiment-correlation-3.png)<!-- -->

```
## 
## [[4]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](world_files/figure-html/sentiment-correlation-4.png)<!-- -->

```
## 
## [[5]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](world_files/figure-html/sentiment-correlation-5.png)<!-- -->

```
## 
## [[6]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](world_files/figure-html/sentiment-correlation-6.png)<!-- -->


We also believe that shares might increase based on whether the post was created on a weekend. Perhaps, weekend posts are shared more frequently as, generally, people have more screen time and thus are more apt to share article. This section `shares` has been scaled and density plotted by `is_weekend` as a factor where 0 is a weekday and 1 is a weekend day.

```r
# Removing scientific notation for readability
options(scipen = 999)

# Coverts is_weekend into a two level factor
weekend_factor <- subset_data_wo_outliers %>%
  mutate(weekend = as.factor(is_weekend)) %>% select(weekend, shares)

# Base plot
g <- ggplot(weekend_factor, aes(x=shares)) +  xlab("Number of Shares") +
  ylab("Density")

# Filled, stacked histogram with density of shares by weekend level
g + geom_histogram(aes(y = ..density.., fill = weekend)) + 
  geom_density(adjust = 0.25, alpha = 0.5, aes(fill = weekend), position = "stack") +
  labs(title = "Density of Shares: Weekday vs. Weekend") + 
  scale_fill_discrete(name = "Weekday or Weekend?", labels = c("Weekday", "Weekend"))
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

![](world_files/figure-html/shares-weekend-plot-1.png)<!-- -->

```r
# Turning on scientific notation
options(scipen = 0)
```

# Modeling

## Linear Models

Fitting our linear model with our chosen variables. 
In this first linear model we will investigate our suposition that readability of the title and content increase shares.

```r
# First linear model
lmfit1 <- lm(shares ~kw_avg_avg*average_token_length*n_tokens_content*n_tokens_title, data = training_data)
summary(lmfit1)
```

```
## 
## Call:
## lm(formula = shares ~ kw_avg_avg * average_token_length * n_tokens_content * 
##     n_tokens_title, data = training_data)
## 
## Residuals:
##    Min     1Q Median     3Q    Max 
##  -8847  -1343   -890   -226  93858 
## 
## Coefficients:
##                                                                   Estimate
## (Intercept)                                                     -1.722e+04
## kw_avg_avg                                                       6.799e+00
## average_token_length                                             3.641e+03
## n_tokens_content                                                 9.397e+00
## n_tokens_title                                                   1.814e+03
## kw_avg_avg:average_token_length                                 -1.375e+00
## kw_avg_avg:n_tokens_content                                     -2.959e-03
## average_token_length:n_tokens_content                           -2.402e+00
## kw_avg_avg:n_tokens_title                                       -5.698e-01
## average_token_length:n_tokens_title                             -3.666e+02
## n_tokens_content:n_tokens_title                                 -1.558e+00
## kw_avg_avg:average_token_length:n_tokens_content                 8.481e-04
## kw_avg_avg:average_token_length:n_tokens_title                   1.259e-01
## kw_avg_avg:n_tokens_content:n_tokens_title                       7.364e-04
## average_token_length:n_tokens_content:n_tokens_title             3.683e-01
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title -1.768e-04
##                                                                 Std. Error
## (Intercept)                                                      6.682e+03
## kw_avg_avg                                                       2.287e+00
## average_token_length                                             1.449e+03
## n_tokens_content                                                 3.024e+01
## n_tokens_title                                                   6.161e+02
## kw_avg_avg:average_token_length                                  5.010e-01
## kw_avg_avg:n_tokens_content                                      1.107e-02
## average_token_length:n_tokens_content                            6.379e+00
## kw_avg_avg:n_tokens_title                                        2.102e-01
## average_token_length:n_tokens_title                              1.337e+02
## n_tokens_content:n_tokens_title                                  2.811e+00
## kw_avg_avg:average_token_length:n_tokens_content                 2.346e-03
## kw_avg_avg:average_token_length:n_tokens_title                   4.604e-02
## kw_avg_avg:n_tokens_content:n_tokens_title                       1.038e-03
## average_token_length:n_tokens_content:n_tokens_title             5.961e-01
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title  2.213e-04
##                                                                 t value
## (Intercept)                                                      -2.576
## kw_avg_avg                                                        2.972
## average_token_length                                              2.512
## n_tokens_content                                                  0.311
## n_tokens_title                                                    2.944
## kw_avg_avg:average_token_length                                  -2.745
## kw_avg_avg:n_tokens_content                                      -0.267
## average_token_length:n_tokens_content                            -0.377
## kw_avg_avg:n_tokens_title                                        -2.711
## average_token_length:n_tokens_title                              -2.742
## n_tokens_content:n_tokens_title                                  -0.554
## kw_avg_avg:average_token_length:n_tokens_content                  0.361
## kw_avg_avg:average_token_length:n_tokens_title                    2.736
## kw_avg_avg:n_tokens_content:n_tokens_title                        0.709
## average_token_length:n_tokens_content:n_tokens_title              0.618
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title  -0.799
##                                                                 Pr(>|t|)
## (Intercept)                                                      0.01001
## kw_avg_avg                                                       0.00297
## average_token_length                                             0.01202
## n_tokens_content                                                 0.75598
## n_tokens_title                                                   0.00325
## kw_avg_avg:average_token_length                                  0.00606
## kw_avg_avg:n_tokens_content                                      0.78918
## average_token_length:n_tokens_content                            0.70651
## kw_avg_avg:n_tokens_title                                        0.00673
## average_token_length:n_tokens_title                              0.00612
## n_tokens_content:n_tokens_title                                  0.57931
## kw_avg_avg:average_token_length:n_tokens_content                 0.71774
## kw_avg_avg:average_token_length:n_tokens_title                   0.00624
## kw_avg_avg:n_tokens_content:n_tokens_title                       0.47813
## average_token_length:n_tokens_content:n_tokens_title             0.53664
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title  0.42442
##                                                                   
## (Intercept)                                                     * 
## kw_avg_avg                                                      **
## average_token_length                                            * 
## n_tokens_content                                                  
## n_tokens_title                                                  **
## kw_avg_avg:average_token_length                                 **
## kw_avg_avg:n_tokens_content                                       
## average_token_length:n_tokens_content                             
## kw_avg_avg:n_tokens_title                                       **
## average_token_length:n_tokens_title                             **
## n_tokens_content:n_tokens_title                                   
## kw_avg_avg:average_token_length:n_tokens_content                  
## kw_avg_avg:average_token_length:n_tokens_title                  **
## kw_avg_avg:n_tokens_content:n_tokens_title                        
## average_token_length:n_tokens_content:n_tokens_title              
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title   
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 4646 on 5882 degrees of freedom
## Multiple R-squared:  0.02243,	Adjusted R-squared:  0.01994 
## F-statistic: 8.999 on 15 and 5882 DF,  p-value: < 2.2e-16
```

Using the `data_source_is_lifestyle` as base data source, the following chunk of code was run and to determine potential significant additive and interaction terms by t tests for significance with p-values less than or equal to 0.05. The subsequent model was selected for this report:
`lm(shares ~  n_tokens_content*num_imgs + n_tokens_content*num_videos + 
                n_tokens_content:average_token_length + n_tokens_content*kw_avg_avg + 
                n_tokens_content*global_sentiment_polarity + 
                n_tokens_content*global_rate_negative_words + num_imgs*kw_avg_avg + 
                num_imgs*abs_title_sentiment_polarity + num_videos*average_token_length + 
                num_videos*global_subjectivity + num_videos*global_sentiment_polarity + 
                num_videos*global_rate_negative_words + num_videos*abs_title_sentiment_polarity,
              data = training_data)`

```r
# Generates a linear model with all main effects and interactions terms
lm_full <- lm(shares ~ .^2, data = training_data)
summary(lm_full)
```

This is the second linear model we will use as described above it was selected by examining p-values from the summary of the a linear model with all main effects and interaction terms.

```r
# Generalates second linear model candidate
lmfit2 <- lm(shares ~  n_tokens_content*num_imgs + n_tokens_content*num_videos + 
                n_tokens_content:average_token_length + n_tokens_content*kw_avg_avg + 
                n_tokens_content*global_sentiment_polarity + 
                n_tokens_content*global_rate_negative_words + num_imgs*kw_avg_avg + 
                num_imgs*abs_title_sentiment_polarity + num_videos*average_token_length + 
                num_videos*global_subjectivity + num_videos*global_sentiment_polarity + 
                num_videos*global_rate_negative_words + num_videos*abs_title_sentiment_polarity,
              data = training_data)
summary(lmfit2)
```

```
## 
## Call:
## lm(formula = shares ~ n_tokens_content * num_imgs + n_tokens_content * 
##     num_videos + n_tokens_content:average_token_length + n_tokens_content * 
##     kw_avg_avg + n_tokens_content * global_sentiment_polarity + 
##     n_tokens_content * global_rate_negative_words + num_imgs * 
##     kw_avg_avg + num_imgs * abs_title_sentiment_polarity + num_videos * 
##     average_token_length + num_videos * global_subjectivity + 
##     num_videos * global_sentiment_polarity + num_videos * global_rate_negative_words + 
##     num_videos * abs_title_sentiment_polarity, data = training_data)
## 
## Residuals:
##    Min     1Q Median     3Q    Max 
##  -9095  -1375   -822   -130  93587 
## 
## Coefficients:
##                                               Estimate Std. Error t value
## (Intercept)                                  1.855e+03  5.687e+02   3.262
## n_tokens_content                             3.792e+00  2.066e+00   1.836
## num_imgs                                    -1.046e+01  4.587e+01  -0.228
## num_videos                                   6.096e+02  5.687e+02   1.072
## kw_avg_avg                                   3.375e-01  1.278e-01   2.640
## global_sentiment_polarity                    2.383e+03  1.651e+03   1.443
## global_rate_negative_words                   1.694e+04  1.525e+04   1.111
## abs_title_sentiment_polarity                -1.004e+02  3.639e+02  -0.276
## average_token_length                        -6.055e+02  1.268e+02  -4.775
## global_subjectivity                          4.402e+03  8.714e+02   5.051
## n_tokens_content:num_imgs                   -3.274e-02  1.596e-02  -2.052
## n_tokens_content:num_videos                  2.865e-02  6.900e-02   0.415
## n_tokens_content:average_token_length       -6.886e-01  4.073e-01  -1.691
## n_tokens_content:kw_avg_avg                 -2.077e-05  1.735e-04  -0.120
## n_tokens_content:global_sentiment_polarity  -3.401e+00  2.659e+00  -1.279
## n_tokens_content:global_rate_negative_words -2.263e+01  2.501e+01  -0.905
## num_imgs:kw_avg_avg                          3.621e-02  1.316e-02   2.752
## num_imgs:abs_title_sentiment_polarity        4.921e+01  5.291e+01   0.930
## num_videos:average_token_length             -7.454e+01  1.145e+02  -0.651
## num_videos:global_subjectivity              -9.662e+00  5.532e+02  -0.017
## num_videos:global_sentiment_polarity        -8.137e+02  6.807e+02  -1.195
## num_videos:global_rate_negative_words       -5.565e+03  6.162e+03  -0.903
## num_videos:abs_title_sentiment_polarity      2.659e+02  1.663e+02   1.599
##                                             Pr(>|t|)    
## (Intercept)                                  0.00111 ** 
## n_tokens_content                             0.06646 .  
## num_imgs                                     0.81961    
## num_videos                                   0.28382    
## kw_avg_avg                                   0.00831 ** 
## global_sentiment_polarity                    0.14894    
## global_rate_negative_words                   0.26676    
## abs_title_sentiment_polarity                 0.78252    
## average_token_length                        1.84e-06 ***
## global_subjectivity                         4.52e-07 ***
## n_tokens_content:num_imgs                    0.04022 *  
## n_tokens_content:num_videos                  0.67797    
## n_tokens_content:average_token_length        0.09096 .  
## n_tokens_content:kw_avg_avg                  0.90473    
## n_tokens_content:global_sentiment_polarity   0.20099    
## n_tokens_content:global_rate_negative_words  0.36556    
## num_imgs:kw_avg_avg                          0.00595 ** 
## num_imgs:abs_title_sentiment_polarity        0.35240    
## num_videos:average_token_length              0.51511    
## num_videos:global_subjectivity               0.98606    
## num_videos:global_sentiment_polarity         0.23203    
## num_videos:global_rate_negative_words        0.36657    
## num_videos:abs_title_sentiment_polarity      0.10981    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 4619 on 5875 degrees of freedom
## Multiple R-squared:  0.03505,	Adjusted R-squared:  0.03144 
## F-statistic: 9.701 on 22 and 5875 DF,  p-value: < 2.2e-16
```

With a simple linear model we can test it's goodness of fit with $R^2$. Since we are measuring human behavior we can be comfortable with a low $R^2$. However too low (although subjective) would indicate that our hypothesis is wrong. As a rule of thumb we will say:


$$R^2 < .1 : we \space suspect \space that \space we \space cannot \space reject \space H_0 \\
  R^2 \space between \space .1 \space and \space .5 \space : \space we \space suspect \space that \space we \space would \space reject \space H_0 \space \\
  R^2 > .5 \space we \space feel \space confident \space that \space our \space variables \space are \space good \space predictors \space and \space our \space hypothesis \space is \space a \space good \space explanation.$$

## Ensemble Methods

### Random Forest
We can now fit our model above into a tree function. This will give us a better picture of where our variables are most important in our model.

```r
fitTree <- tree(shares ~kw_avg_avg + average_token_length + n_tokens_content + n_tokens_title, data = training_data)
plot(fitTree)
text(fitTree)
```

![](world_files/figure-html/fitting-tree-model-1.png)<!-- -->

We are able to use the tree function to see where our variables are most important and in what order. This could change based on subject.

We can kick off a random forest model in our to see if adding this level of complexity for our model is needed/beneficial.

```r
#Random Forest
#Train control options for ensemble models
trCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

rfFit <- train(shares ~kw_avg_avg + average_token_length + n_tokens_content + n_tokens_title, data = training_data, method = "rf",trControl=trCtrl, preProcess = c("center", "scale"),tuneGrid = data.frame(mtry = 1:4))

plot(rfFit)
```

![](world_files/figure-html/random-forest-1.png)<!-- -->

By plotting the `rfFit` we can see which `mtry` value is the best. This might be different between subjects.


### Boosted Tree
Lastly, a boosted tree was fit using 5-fold, 3 times repeated cross-validation with tuning parameter combinations of `n.trees` = (10, 25, 50, 100, 150, and 200), `interaction.depth` = 1:4, `shrinkage` = 0.1, and `n.minobsinnode` = 10.

```r
#Tune Parameters for cross-validation
tuneParam <-expand.grid(n.trees = c(10, 25, 50, 100, 150, 200),
                         interaction.depth = 1:4, shrinkage = 0.1,
                         n.minobsinnode = 10)
#Cross-validation, tune parameter selection, and training of boosted tree models
boostTree <-train(shares ~ ., data = training_data, method = "gbm",
                trControl = trCtrl, preProcess = c("center","scale"),
                tuneGrid = tuneParam)
```

Here are results of the cross-validation for selecting the best model.

```r
#Plot of RMSE by Max Tree Depth
plot(boostTree)
```

![](world_files/figure-html/boosted-tree-results-1.png)<!-- -->

```r
#Results from model training
boostTree$results
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["shrinkage"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["interaction.depth"],"name":[2],"type":["int"],"align":["right"]},{"label":["n.minobsinnode"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["n.trees"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["RMSE"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["Rsquared"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["MAE"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["RMSESD"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["RsquaredSD"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["MAESD"],"name":[10],"type":["dbl"],"align":["right"]}],"data":[{"1":"0.1","2":"1","3":"10","4":"10","5":"4598.204","6":"0.02325666","7":"1827.113","8":"764.8197","9":"0.01222106","10":"94.57021","_rn_":"1"},{"1":"0.1","2":"2","3":"10","4":"10","5":"4597.489","6":"0.02039781","7":"1818.354","8":"765.1822","9":"0.01178596","10":"96.31958","_rn_":"7"},{"1":"0.1","2":"3","3":"10","4":"10","5":"4600.432","6":"0.01803806","7":"1818.743","8":"763.5836","9":"0.01233260","10":"99.16712","_rn_":"13"},{"1":"0.1","2":"4","3":"10","4":"10","5":"4597.659","6":"0.01936325","7":"1819.793","8":"770.3333","9":"0.01550094","10":"99.57427","_rn_":"19"},{"1":"0.1","2":"1","3":"10","4":"25","5":"4576.456","6":"0.02905779","7":"1798.241","8":"766.9682","9":"0.01277037","10":"94.13710","_rn_":"2"},{"1":"0.1","2":"2","3":"10","4":"25","5":"4590.267","6":"0.02237543","7":"1803.830","8":"765.5610","9":"0.01216282","10":"96.66949","_rn_":"8"},{"1":"0.1","2":"3","3":"10","4":"25","5":"4601.749","6":"0.02071199","7":"1804.897","8":"764.0153","9":"0.01357923","10":"98.00521","_rn_":"14"},{"1":"0.1","2":"4","3":"10","4":"25","5":"4595.244","6":"0.02541343","7":"1809.491","8":"775.3910","9":"0.02106836","10":"103.19448","_rn_":"20"},{"1":"0.1","2":"1","3":"10","4":"50","5":"4569.063","6":"0.03064639","7":"1789.851","8":"765.9300","9":"0.01182157","10":"98.51375","_rn_":"3"},{"1":"0.1","2":"2","3":"10","4":"50","5":"4601.996","6":"0.02365218","7":"1802.319","8":"754.9290","9":"0.01114720","10":"95.42716","_rn_":"9"},{"1":"0.1","2":"3","3":"10","4":"50","5":"4626.367","6":"0.02262689","7":"1811.368","8":"761.7161","9":"0.01427294","10":"106.27616","_rn_":"15"},{"1":"0.1","2":"4","3":"10","4":"50","5":"4630.365","6":"0.02402446","7":"1820.373","8":"765.0166","9":"0.01603477","10":"107.49168","_rn_":"21"},{"1":"0.1","2":"1","3":"10","4":"100","5":"4570.048","6":"0.03235113","7":"1786.935","8":"766.3623","9":"0.01272159","10":"98.34612","_rn_":"4"},{"1":"0.1","2":"2","3":"10","4":"100","5":"4620.239","6":"0.02464070","7":"1810.756","8":"754.4338","9":"0.01239989","10":"97.09388","_rn_":"10"},{"1":"0.1","2":"3","3":"10","4":"100","5":"4654.531","6":"0.02295570","7":"1825.824","8":"759.4734","9":"0.01534789","10":"106.97605","_rn_":"16"},{"1":"0.1","2":"4","3":"10","4":"100","5":"4670.644","6":"0.02177824","7":"1849.470","8":"760.1642","9":"0.01419581","10":"103.76884","_rn_":"22"},{"1":"0.1","2":"1","3":"10","4":"150","5":"4572.872","6":"0.03247245","7":"1791.800","8":"767.7791","9":"0.01316274","10":"98.59445","_rn_":"5"},{"1":"0.1","2":"2","3":"10","4":"150","5":"4640.967","6":"0.02234519","7":"1822.340","8":"754.4403","9":"0.01106682","10":"97.55438","_rn_":"11"},{"1":"0.1","2":"3","3":"10","4":"150","5":"4692.135","6":"0.01957304","7":"1854.983","8":"744.1297","9":"0.01294834","10":"99.95148","_rn_":"17"},{"1":"0.1","2":"4","3":"10","4":"150","5":"4712.477","6":"0.01907359","7":"1877.895","8":"748.8436","9":"0.01137914","10":"99.29862","_rn_":"23"},{"1":"0.1","2":"1","3":"10","4":"200","5":"4572.463","6":"0.03308732","7":"1796.152","8":"765.0555","9":"0.01272351","10":"97.64064","_rn_":"6"},{"1":"0.1","2":"2","3":"10","4":"200","5":"4657.513","6":"0.02154420","7":"1839.265","8":"747.6192","9":"0.01253423","10":"95.18227","_rn_":"12"},{"1":"0.1","2":"3","3":"10","4":"200","5":"4714.954","6":"0.01820262","7":"1878.042","8":"739.5343","9":"0.01181640","10":"99.24679","_rn_":"18"},{"1":"0.1","2":"4","3":"10","4":"200","5":"4744.652","6":"0.01722521","7":"1910.119","8":"743.7001","9":"0.01166431","10":"98.09399","_rn_":"24"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

The best tuning paramters are:

```r
#Best tuning parameters
boostTree$bestTune
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["n.trees"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["interaction.depth"],"name":[2],"type":["int"],"align":["right"]},{"label":["shrinkage"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["n.minobsinnode"],"name":[4],"type":["dbl"],"align":["right"]}],"data":[{"1":"50","2":"1","3":"0.1","4":"10","_rn_":"3"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

This section generates the predictions based on the best boosted tree model. 

```r
#Uses best tuned training boosted tree model to predict test data
boostTreePred <-predict(boostTree, newdata = test_data)

#Reports best boosted tree model and corresponding RMSE
boost <-boostTree$results[1,]
boost
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["shrinkage"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["interaction.depth"],"name":[2],"type":["int"],"align":["right"]},{"label":["n.minobsinnode"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["n.trees"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["RMSE"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["Rsquared"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["MAE"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["RMSESD"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["RsquaredSD"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["MAESD"],"name":[10],"type":["dbl"],"align":["right"]}],"data":[{"1":"0.1","2":"1","3":"10","4":"10","5":"4598.204","6":"0.02325666","7":"1827.113","8":"764.8197","9":"0.01222106","10":"94.57021","_rn_":"1"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

# Comparison


```r
lmfit_1 = c(MAE(test_data$shares,predict(lmfit1)),RMSE(test_data$shares,predict(lmfit1)))

lmfit_2 = c(MAE(test_data$shares,predict(lmfit2)),RMSE(test_data$shares,predict(lmfit2)))

rffit_c = c(MAE(test_data$shares,predict(rfFit)),RMSE(test_data$shares,predict(rfFit)))

boostTree_c = c(MAE(test_data$shares,predict(boostTree)),RMSE(test_data$shares,predict(boostTree)))

MAE_RMSE_SUMM <- rbind.data.frame("Linear Model 1" = lmfit_1, "Linear Model 2" = lmfit_2,"Random Forrest" = rffit_c, "Boosted Tree" = boostTree_c)

colnames(MAE_RMSE_SUMM) <- c("MAE","RMSE")
rownames(MAE_RMSE_SUMM) <- c("Linear Model 1", "Linear Model 2", "Random Forrest", "Boosted Tree")
kable(MAE_RMSE_SUMM, caption = "Comparing models via MAE and RMSE")
```



Table: Comparing models via MAE and RMSE

|               |      MAE|     RMSE|
|:--------------|--------:|--------:|
|Linear Model 1 | 2111.061| 8352.808|
|Linear Model 2 | 2132.292| 8361.778|
|Random Forrest | 2258.787| 8606.636|
|Boosted Tree   | 2114.559| 8359.037|

Measuring success with MAE and RMSE. This helps measure the models success while also accounting for the complexity of the model. We can use the MAE **and** RMSE to see if any issues in our data come form a single point (or a small subset of points).

# Automation

The following code chunk was called in the console in order to generate a report of each category of data_channel_is_* (`lifestyle`, `entertainment`, `bus`, `socmed`, `tech`, `world`).

```r
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
apply(reports, MARGIN = 1, FUN = function(x){ render(input = "Project 2.Rmd",
                                                     output_file = x[[1]],
                                                     params = x[[2]])})
```
