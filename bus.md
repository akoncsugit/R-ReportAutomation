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
We will subset the data based on the category listed in our YAML header. In this case, using data from `data_channel_is_bus`. We will remove non-predictors such as `url` and `timedelta` and selected our desired predictors** and `shares`.
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
##  n_tokens_title  n_tokens_content    num_imgs       num_videos     
##  Min.   : 3.00   Min.   :   0.0   Min.   : 0.00   Min.   : 0.0000  
##  1st Qu.: 9.00   1st Qu.: 244.0   1st Qu.: 1.00   1st Qu.: 0.0000  
##  Median :10.00   Median : 403.0   Median : 1.00   Median : 0.0000  
##  Mean   :10.29   Mean   : 546.3   Mean   : 1.78   Mean   : 0.6297  
##  3rd Qu.:12.00   3rd Qu.: 728.2   3rd Qu.: 1.00   3rd Qu.: 0.0000  
##  Max.   :19.00   Max.   :6336.0   Max.   :51.00   Max.   :74.0000  
##  average_token_length   kw_avg_avg      is_weekend     
##  Min.   :0.000        Min.   :    0   Min.   :0.00000  
##  1st Qu.:4.527        1st Qu.: 2332   1st Qu.:0.00000  
##  Median :4.688        Median : 2764   Median :0.00000  
##  Mean   :4.687        Mean   : 2955   Mean   :0.09452  
##  3rd Qu.:4.855        3rd Qu.: 3336   3rd Qu.:0.00000  
##  Max.   :5.882        Max.   :43568   Max.   :1.00000  
##  global_subjectivity global_sentiment_polarity global_rate_negative_words
##  Min.   :0.0000      Min.   :-0.23929          Min.   :0.000000          
##  1st Qu.:0.3873      1st Qu.: 0.08671          1st Qu.:0.009091          
##  Median :0.4392      Median : 0.13617          Median :0.014085          
##  Mean   :0.4357      Mean   : 0.13612          Mean   :0.014777          
##  3rd Qu.:0.4886      3rd Qu.: 0.18613          3rd Qu.:0.019373          
##  Max.   :0.8489      Max.   : 0.62258          Max.   :0.060811          
##  avg_negative_polarity abs_title_subjectivity abs_title_sentiment_polarity
##  Min.   :-1.0000       Min.   :0.0000         Min.   :0.0000              
##  1st Qu.:-0.3009       1st Qu.:0.1500         1st Qu.:0.0000              
##  Median :-0.2366       Median :0.5000         Median :0.0000              
##  Mean   :-0.2417       Mean   :0.3386         Mean   :0.1418              
##  3rd Qu.:-0.1766       3rd Qu.:0.5000         3rd Qu.:0.2438              
##  Max.   : 0.0000       Max.   :0.5000         Max.   :1.0000              
##      shares        
##  Min.   :     1.0  
##  1st Qu.:   956.8  
##  Median :  1400.0  
##  Mean   :  3146.0  
##  3rd Qu.:  2500.0  
##  Max.   :690400.0
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
{"columns":[{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"2.15579","2":"459.2981","3":"3.42196","4":"3.3632","5":"0.38714","6":"1533.162","7":"0.29258","8":"0.08296","9":"0.08191","10":"0.00862","11":"0.11146","12":"0.19073","13":"0.20877","14":"14601.32"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
{"columns":[{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"3","2":"484.25","3":"0","4":"0","5":"0.3275305","6":"1004.621","7":"0","8":"0.1012951","9":"0.09942531","10":"0.0102819","11":"0.1242593","12":"0.35","13":"0.2438447","14":"1543.25"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"1.000","2":"-0.008","3":"-0.018","4":"0.039","5":"-0.079","6":"0.016","7":"0.007","8":"0.008","9":"-0.026","10":"0.024","11":"-0.021","12":"-0.181","13":"0.085","14":"0.018","_rn_":"n_tokens_title"},{"1":"-0.008","2":"1.000","3":"0.245","4":"0.161","5":"0.018","6":"0.038","7":"0.178","8":"0.180","9":"0.070","10":"0.100","11":"-0.152","12":"0.010","13":"0.032","14":"0.042","_rn_":"n_tokens_content"},{"1":"-0.018","2":"0.245","3":"1.000","4":"-0.012","5":"0.027","6":"0.033","7":"0.018","8":"0.060","9":"0.035","10":"-0.012","11":"-0.020","12":"0.013","13":"-0.012","14":"0.044","_rn_":"num_imgs"},{"1":"0.039","2":"0.161","3":"-0.012","4":"1.000","5":"-0.031","6":"0.102","7":"0.019","8":"0.060","9":"0.024","10":"0.034","11":"-0.102","12":"-0.047","13":"0.060","14":"0.068","_rn_":"num_videos"},{"1":"-0.079","2":"0.018","3":"0.027","4":"-0.031","5":"1.000","6":"-0.058","7":"0.024","8":"0.147","9":"0.125","10":"-0.050","11":"-0.032","12":"-0.031","13":"-0.032","14":"-0.015","_rn_":"average_token_length"},{"1":"0.016","2":"0.038","3":"0.033","4":"0.102","5":"-0.058","6":"1.000","7":"0.085","8":"0.102","9":"0.030","10":"0.047","11":"-0.104","12":"-0.007","13":"0.060","14":"0.095","_rn_":"kw_avg_avg"},{"1":"0.007","2":"0.178","3":"0.018","4":"0.019","5":"0.024","6":"0.085","7":"1.000","8":"0.054","9":"0.042","10":"0.028","11":"-0.050","12":"-0.009","13":"0.038","14":"0.024","_rn_":"is_weekend"},{"1":"0.008","2":"0.180","3":"0.060","4":"0.060","5":"0.147","6":"0.102","7":"0.054","8":"1.000","9":"0.295","10":"0.104","11":"-0.299","12":"-0.002","13":"0.091","14":"0.042","_rn_":"global_subjectivity"},{"1":"-0.026","2":"0.070","3":"0.035","4":"0.024","5":"0.125","6":"0.030","7":"0.042","8":"0.295","9":"1.000","10":"-0.512","11":"0.286","12":"-0.090","13":"0.098","14":"-0.007","_rn_":"global_sentiment_polarity"},{"1":"0.024","2":"0.100","3":"-0.012","4":"0.034","5":"-0.050","6":"0.047","7":"0.028","8":"0.104","9":"-0.512","10":"1.000","11":"-0.307","12":"0.002","13":"0.043","14":"0.042","_rn_":"global_rate_negative_words"},{"1":"-0.021","2":"-0.152","3":"-0.020","4":"-0.102","5":"-0.032","6":"-0.104","7":"-0.050","8":"-0.299","9":"0.286","10":"-0.307","11":"1.000","12":"-0.024","13":"-0.076","14":"-0.046","_rn_":"avg_negative_polarity"},{"1":"-0.181","2":"0.010","3":"0.013","4":"-0.047","5":"-0.031","6":"-0.007","7":"-0.009","8":"-0.002","9":"-0.090","10":"0.002","11":"-0.024","12":"1.000","13":"-0.500","14":"0.014","_rn_":"abs_title_subjectivity"},{"1":"0.085","2":"0.032","3":"-0.012","4":"0.060","5":"-0.032","6":"0.060","7":"0.038","8":"0.091","9":"0.098","10":"0.043","11":"-0.076","12":"-0.500","13":"1.000","14":"0.023","_rn_":"abs_title_sentiment_polarity"},{"1":"0.018","2":"0.042","3":"0.044","4":"0.068","5":"-0.015","6":"0.095","7":"0.024","8":"0.042","9":"-0.007","10":"0.042","11":"-0.046","12":"0.014","13":"0.023","14":"1.000","_rn_":"shares"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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

![](bus_files/figure-html/boxplot-outliers-1.png)<!-- -->

```r
boxplot(training_data$shares,horizontal = TRUE, range = 2, outline = FALSE,main = "Boxplot of shares without outliers")
```

![](bus_files/figure-html/boxplot-outliers-2.png)<!-- -->

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

![](bus_files/figure-html/shares-vs-keywords-average-1.png)<!-- -->

We can measure the trend of shares as a function of Average number of key words. If we see a possitive trend we can say that the more key words in the articles the more likely it is to be shared, the opposite can also be said. We measure the correlation to get a more precise gauge in case the graph is not clear enough.

```r
correlation2 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$average_token_length)

plot2 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = average_token_length)) +
geom_density_2d() + 
  labs(title = "number of shares vs. Average length of words in content", y= "# of shares", x = "Average length of words in content") +
  geom_text(color = "red",x=5,y=3500,label = paste0("Correlation = ",round(correlation2,3)))

plot2
```

![](bus_files/figure-html/shares-vs-average-length-of-words-in-content-1.png)<!-- -->

With a density plot as a function of average length of words in content we see where most of our shares come from. We can utilize this to help explain our model down below.


```r
correlation3 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$n_tokens_content)

plot3 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = n_tokens_content)) +
geom_rug() +
  labs(title = "number of shares vs. number of words in content", y= "# of shares", x = "# of words in content") +
  geom_text(color = "red",x=4000,y=4000,label = paste0("Correlation = ",round(correlation3,3)))

plot3
```

![](bus_files/figure-html/density-plot-1.png)<!-- -->

Using a rug graph we can measure the relationship between number of words in content and the number of shares. The intersection between where both rugs are highly concentrated is where how we can measure correlation. If both rugs are concentrated near zero than we see that the less words the more shareable the articles are or vice versa.


```r
correlation4 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$n_tokens_title)

plot4 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = n_tokens_title)) +
geom_col() +
  labs(title = "number of shares vs. number of words in title", y= "# of shares", x = "# of words in title") +
  geom_text(color = "red",x=15,y=600000,label = paste0("Correlation = ",round(correlation4,3)))

plot4
```

![](bus_files/figure-html/rug-graph-1.png)<!-- -->
We see how the `# of words in title` as distributed with respect to number of shares. Any large skewness would be a flag for us to research further.


Here we graphically depict the correlations among the variables in the training data.

```r
# Note need to probably name graphs still etc...work in progress
corrplot(cor(training_data), tl.col = "black")
```

![](bus_files/figure-html/correlation-plot-1.png)<!-- -->

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

![](bus_files/figure-html/sentiment-correlation-1.png)<!-- -->

```
## 
## [[2]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](bus_files/figure-html/sentiment-correlation-2.png)<!-- -->

```
## 
## [[3]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](bus_files/figure-html/sentiment-correlation-3.png)<!-- -->

```
## 
## [[4]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](bus_files/figure-html/sentiment-correlation-4.png)<!-- -->

```
## 
## [[5]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](bus_files/figure-html/sentiment-correlation-5.png)<!-- -->

```
## 
## [[6]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](bus_files/figure-html/sentiment-correlation-6.png)<!-- -->


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

![](bus_files/figure-html/shares-weekend-plot-1.png)<!-- -->

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
## -27946  -1961  -1285   -315 683989 
## 
## Coefficients:
##                                                                   Estimate
## (Intercept)                                                      1.675e+04
## kw_avg_avg                                                      -4.530e+00
## average_token_length                                            -3.372e+03
## n_tokens_content                                                 1.301e+02
## n_tokens_title                                                  -1.303e+03
## kw_avg_avg:average_token_length                                  8.796e-01
## kw_avg_avg:n_tokens_content                                     -5.428e-02
## average_token_length:n_tokens_content                           -2.691e+01
## kw_avg_avg:n_tokens_title                                        4.143e-01
## average_token_length:n_tokens_title                              2.768e+02
## n_tokens_content:n_tokens_title                                 -1.002e+01
## kw_avg_avg:average_token_length:n_tokens_content                 1.163e-02
## kw_avg_avg:average_token_length:n_tokens_title                  -6.824e-02
## kw_avg_avg:n_tokens_content:n_tokens_title                       4.637e-03
## average_token_length:n_tokens_content:n_tokens_title             2.027e+00
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title -9.770e-04
##                                                                 Std. Error
## (Intercept)                                                      3.436e+04
## kw_avg_avg                                                       1.027e+01
## average_token_length                                             7.503e+03
## n_tokens_content                                                 1.157e+02
## n_tokens_title                                                   3.237e+03
## kw_avg_avg:average_token_length                                  2.266e+00
## kw_avg_avg:n_tokens_content                                      3.503e-02
## average_token_length:n_tokens_content                            2.517e+01
## kw_avg_avg:n_tokens_title                                        9.790e-01
## average_token_length:n_tokens_title                              7.149e+02
## n_tokens_content:n_tokens_title                                  1.104e+01
## kw_avg_avg:average_token_length:n_tokens_content                 7.635e-03
## kw_avg_avg:average_token_length:n_tokens_title                   2.184e-01
## kw_avg_avg:n_tokens_content:n_tokens_title                       3.362e-03
## average_token_length:n_tokens_content:n_tokens_title             2.408e+00
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title  7.341e-04
##                                                                 t value
## (Intercept)                                                       0.487
## kw_avg_avg                                                       -0.441
## average_token_length                                             -0.449
## n_tokens_content                                                  1.124
## n_tokens_title                                                   -0.402
## kw_avg_avg:average_token_length                                   0.388
## kw_avg_avg:n_tokens_content                                      -1.550
## average_token_length:n_tokens_content                            -1.069
## kw_avg_avg:n_tokens_title                                         0.423
## average_token_length:n_tokens_title                               0.387
## n_tokens_content:n_tokens_title                                  -0.908
## kw_avg_avg:average_token_length:n_tokens_content                  1.523
## kw_avg_avg:average_token_length:n_tokens_title                   -0.312
## kw_avg_avg:n_tokens_content:n_tokens_title                        1.379
## average_token_length:n_tokens_content:n_tokens_title              0.842
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title  -1.331
##                                                                 Pr(>|t|)
## (Intercept)                                                        0.626
## kw_avg_avg                                                         0.659
## average_token_length                                               0.653
## n_tokens_content                                                   0.261
## n_tokens_title                                                     0.687
## kw_avg_avg:average_token_length                                    0.698
## kw_avg_avg:n_tokens_content                                        0.121
## average_token_length:n_tokens_content                              0.285
## kw_avg_avg:n_tokens_title                                          0.672
## average_token_length:n_tokens_title                                0.699
## n_tokens_content:n_tokens_title                                    0.364
## kw_avg_avg:average_token_length:n_tokens_content                   0.128
## kw_avg_avg:average_token_length:n_tokens_title                     0.755
## kw_avg_avg:n_tokens_content:n_tokens_title                         0.168
## average_token_length:n_tokens_content:n_tokens_title               0.400
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title    0.183
## 
## Residual standard error: 14520 on 4364 degrees of freedom
## Multiple R-squared:  0.01481,	Adjusted R-squared:  0.01142 
## F-statistic: 4.373 on 15 and 4364 DF,  p-value: 3.14e-08
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
## -42481  -1901  -1011    -44 673747 
## 
## Coefficients:
##                                               Estimate Std. Error t value
## (Intercept)                                 -1.162e+03  3.465e+03  -0.335
## n_tokens_content                            -1.871e+00  7.937e+00  -0.236
## num_imgs                                     6.060e+02  2.139e+02   2.834
## num_videos                                   1.435e+03  1.518e+03   0.945
## kw_avg_avg                                   5.726e-01  2.126e-01   2.694
## global_sentiment_polarity                   -7.291e+03  5.032e+03  -1.449
## global_rate_negative_words                   5.206e+03  4.900e+04   0.106
## abs_title_sentiment_polarity                 9.328e+02  1.252e+03   0.745
## average_token_length                         4.912e+01  7.450e+02   0.066
## global_subjectivity                          4.306e+03  3.112e+03   1.384
## n_tokens_content:num_imgs                   -1.786e-01  9.923e-02  -1.800
## n_tokens_content:num_videos                 -4.881e-01  1.142e-01  -4.272
## n_tokens_content:average_token_length       -5.164e-01  1.542e+00  -0.335
## n_tokens_content:kw_avg_avg                  7.853e-04  4.504e-04   1.743
## n_tokens_content:global_sentiment_polarity   1.509e+01  9.461e+00   1.595
## n_tokens_content:global_rate_negative_words  6.408e+01  9.760e+01   0.657
## num_imgs:kw_avg_avg                         -1.023e-01  5.290e-02  -1.934
## num_imgs:abs_title_sentiment_polarity        2.384e+02  3.541e+02   0.673
## num_videos:average_token_length              6.894e+01  3.062e+02   0.225
## num_videos:global_subjectivity              -2.652e+03  1.453e+03  -1.825
## num_videos:global_sentiment_polarity         8.073e+02  1.986e+03   0.406
## num_videos:global_rate_negative_words        3.169e+04  1.475e+04   2.148
## num_videos:abs_title_sentiment_polarity     -1.035e+03  3.534e+02  -2.929
##                                             Pr(>|t|)    
## (Intercept)                                  0.73728    
## n_tokens_content                             0.81365    
## num_imgs                                     0.00462 ** 
## num_videos                                   0.34471    
## kw_avg_avg                                   0.00709 ** 
## global_sentiment_polarity                    0.14746    
## global_rate_negative_words                   0.91538    
## abs_title_sentiment_polarity                 0.45627    
## average_token_length                         0.94744    
## global_subjectivity                          0.16657    
## n_tokens_content:num_imgs                    0.07196 .  
## n_tokens_content:num_videos                 1.98e-05 ***
## n_tokens_content:average_token_length        0.73766    
## n_tokens_content:kw_avg_avg                  0.08132 .  
## n_tokens_content:global_sentiment_polarity   0.11071    
## n_tokens_content:global_rate_negative_words  0.51149    
## num_imgs:kw_avg_avg                          0.05316 .  
## num_imgs:abs_title_sentiment_polarity        0.50087    
## num_videos:average_token_length              0.82188    
## num_videos:global_subjectivity               0.06803 .  
## num_videos:global_sentiment_polarity         0.68446    
## num_videos:global_rate_negative_words        0.03173 *  
## num_videos:abs_title_sentiment_polarity      0.00342 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 14450 on 4357 degrees of freedom
## Multiple R-squared:  0.02544,	Adjusted R-squared:  0.02052 
## F-statistic:  5.17 on 22 and 4357 DF,  p-value: 3.879e-14
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

![](bus_files/figure-html/fitting-tree-model-1.png)<!-- -->

We are able to use the tree function to see where our variables are most important and in what order. This could change based on subject.

We can kick off a random forest model in our to see if adding this level of complexity for our model is needed/beneficial.

```r
#Random Forest
#Train control options for ensemble models
trCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

rfFit <- train(shares ~kw_avg_avg + average_token_length + n_tokens_content + n_tokens_title, data = training_data, method = "rf",trControl=trCtrl, preProcess = c("center", "scale"),tuneGrid = data.frame(mtry = 1:4))

plot(rfFit)
```

![](bus_files/figure-html/random-forest-1.png)<!-- -->

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

![](bus_files/figure-html/boosted-tree-results-1.png)<!-- -->

```r
#Results from model training
boostTree$results
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["shrinkage"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["interaction.depth"],"name":[2],"type":["int"],"align":["right"]},{"label":["n.minobsinnode"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["n.trees"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["RMSE"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["Rsquared"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["MAE"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["RMSESD"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["RsquaredSD"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["MAESD"],"name":[10],"type":["dbl"],"align":["right"]}],"data":[{"1":"0.1","2":"1","3":"10","4":"10","5":"12501.08","6":"0.02040157","7":"2855.906","8":"7704.298","9":"0.01714217","10":"393.7807","_rn_":"1"},{"1":"0.1","2":"2","3":"10","4":"10","5":"12527.34","6":"0.02528350","7":"2834.855","8":"7618.416","9":"0.02313799","10":"395.8642","_rn_":"7"},{"1":"0.1","2":"3","3":"10","4":"10","5":"12617.36","6":"0.01851238","7":"2864.769","8":"7608.975","9":"0.01653405","10":"389.2226","_rn_":"13"},{"1":"0.1","2":"4","3":"10","4":"10","5":"12614.11","6":"0.02171200","7":"2845.341","8":"7543.857","9":"0.01663587","10":"365.1232","_rn_":"19"},{"1":"0.1","2":"1","3":"10","4":"25","5":"12614.22","6":"0.02047106","7":"2866.210","8":"7600.962","9":"0.01429624","10":"367.1414","_rn_":"2"},{"1":"0.1","2":"2","3":"10","4":"25","5":"12806.34","6":"0.02285191","7":"2901.713","8":"7401.429","9":"0.01812992","10":"383.1524","_rn_":"8"},{"1":"0.1","2":"3","3":"10","4":"25","5":"12969.89","6":"0.02051878","7":"2939.301","8":"7309.120","9":"0.01760093","10":"366.0259","_rn_":"14"},{"1":"0.1","2":"4","3":"10","4":"25","5":"12891.18","6":"0.02421845","7":"2909.102","8":"7341.155","9":"0.01973467","10":"374.1626","_rn_":"20"},{"1":"0.1","2":"1","3":"10","4":"50","5":"12697.94","6":"0.01992393","7":"2880.236","8":"7538.260","9":"0.01431630","10":"360.9872","_rn_":"3"},{"1":"0.1","2":"2","3":"10","4":"50","5":"13197.13","6":"0.02110523","7":"3011.092","8":"7128.232","9":"0.01455098","10":"331.7365","_rn_":"9"},{"1":"0.1","2":"3","3":"10","4":"50","5":"13142.39","6":"0.01954490","7":"3030.439","8":"7232.618","9":"0.01745183","10":"389.3769","_rn_":"15"},{"1":"0.1","2":"4","3":"10","4":"50","5":"13189.90","6":"0.01958471","7":"3010.304","8":"7120.920","9":"0.01551389","10":"337.0112","_rn_":"21"},{"1":"0.1","2":"1","3":"10","4":"100","5":"12708.89","6":"0.01930936","7":"2878.213","8":"7510.707","9":"0.01217288","10":"359.3360","_rn_":"4"},{"1":"0.1","2":"2","3":"10","4":"100","5":"13446.00","6":"0.01716701","7":"3112.154","8":"6975.979","9":"0.01209651","10":"322.1092","_rn_":"10"},{"1":"0.1","2":"3","3":"10","4":"100","5":"13519.90","6":"0.01438564","7":"3136.274","8":"7002.921","9":"0.01159098","10":"357.8102","_rn_":"16"},{"1":"0.1","2":"4","3":"10","4":"100","5":"13501.95","6":"0.01551108","7":"3131.895","8":"7024.579","9":"0.01315352","10":"324.6471","_rn_":"22"},{"1":"0.1","2":"1","3":"10","4":"150","5":"12700.59","6":"0.02096438","7":"2894.439","8":"7519.572","9":"0.01451815","10":"367.1632","_rn_":"5"},{"1":"0.1","2":"2","3":"10","4":"150","5":"13793.58","6":"0.01612404","7":"3228.500","8":"6795.128","9":"0.01243728","10":"308.7765","_rn_":"11"},{"1":"0.1","2":"3","3":"10","4":"150","5":"13719.92","6":"0.01416050","7":"3217.108","8":"6905.746","9":"0.01441906","10":"365.1472","_rn_":"17"},{"1":"0.1","2":"4","3":"10","4":"150","5":"13802.17","6":"0.01402827","7":"3237.545","8":"6914.283","9":"0.01448077","10":"336.4503","_rn_":"23"},{"1":"0.1","2":"1","3":"10","4":"200","5":"12728.89","6":"0.02021930","7":"2918.348","8":"7518.484","9":"0.01363483","10":"390.2743","_rn_":"6"},{"1":"0.1","2":"2","3":"10","4":"200","5":"13987.67","6":"0.01408604","7":"3290.297","8":"6704.136","9":"0.01211025","10":"293.8362","_rn_":"12"},{"1":"0.1","2":"3","3":"10","4":"200","5":"13851.13","6":"0.01349312","7":"3272.086","8":"6870.275","9":"0.01300456","10":"339.2289","_rn_":"18"},{"1":"0.1","2":"4","3":"10","4":"200","5":"14044.46","6":"0.01241474","7":"3314.589","8":"6807.723","9":"0.01151728","10":"343.9644","_rn_":"24"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

The best tuning paramters are:

```r
#Best tuning parameters
boostTree$bestTune
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["n.trees"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["interaction.depth"],"name":[2],"type":["int"],"align":["right"]},{"label":["shrinkage"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["n.minobsinnode"],"name":[4],"type":["dbl"],"align":["right"]}],"data":[{"1":"10","2":"1","3":"0.1","4":"10","_rn_":"1"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["shrinkage"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["interaction.depth"],"name":[2],"type":["int"],"align":["right"]},{"label":["n.minobsinnode"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["n.trees"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["RMSE"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["Rsquared"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["MAE"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["RMSESD"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["RsquaredSD"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["MAESD"],"name":[10],"type":["dbl"],"align":["right"]}],"data":[{"1":"0.1","2":"1","3":"10","4":"10","5":"12501.08","6":"0.02040157","7":"2855.906","8":"7704.298","9":"0.01714217","10":"393.7807","_rn_":"1"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
|Linear Model 1 | 2707.329| 15038.54|
|Linear Model 2 | 2820.606| 15107.63|
|Random Forrest | 2962.341| 16607.31|
|Boosted Tree   | 2530.906| 14946.48|

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
