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
We will subset the data based on the category listed in our YAML header. In this case, using data from `data_channel_is_entertainment`. We will remove non-predictors such as `url` and `timedelta` and selected our desired predictors** and `shares`.
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
##  n_tokens_title n_tokens_content    num_imgs         num_videos    
##  Min.   : 2     Min.   :   0.0   Min.   :  0.000   Min.   : 0.000  
##  1st Qu.:10     1st Qu.: 254.0   1st Qu.:  1.000   1st Qu.: 0.000  
##  Median :11     Median : 433.0   Median :  1.000   Median : 1.000  
##  Mean   :11     Mean   : 606.4   Mean   :  6.278   Mean   : 2.636  
##  3rd Qu.:12     3rd Qu.: 802.0   3rd Qu.:  8.000   3rd Qu.: 1.000  
##  Max.   :18     Max.   :6505.0   Max.   :128.000   Max.   :74.000  
##  average_token_length   kw_avg_avg      is_weekend     global_subjectivity
##  Min.   :0.000        Min.   :    0   Min.   :0.0000   Min.   :0.0000     
##  1st Qu.:4.429        1st Qu.: 2551   1st Qu.:0.0000   1st Qu.:0.4116     
##  Median :4.586        Median : 2972   Median :0.0000   Median :0.4632     
##  Mean   :4.475        Mean   : 3152   Mean   :0.1363   Mean   :0.4519     
##  3rd Qu.:4.753        3rd Qu.: 3535   3rd Qu.:0.0000   3rd Qu.:0.5124     
##  Max.   :7.696        Max.   :14138   Max.   :1.0000   Max.   :0.8333     
##  global_sentiment_polarity global_rate_negative_words
##  Min.   :-0.37393          Min.   :0.00000           
##  1st Qu.: 0.04643          1st Qu.:0.01061           
##  Median : 0.11099          Median :0.01712           
##  Mean   : 0.11082          Mean   :0.01905           
##  3rd Qu.: 0.17382          3rd Qu.:0.02492           
##  Max.   : 0.72784          Max.   :0.09104           
##  avg_negative_polarity abs_title_subjectivity abs_title_sentiment_polarity
##  Min.   :-1.0000       Min.   :0.0000         Min.   :0.00000             
##  1st Qu.:-0.3692       1st Qu.:0.1227         1st Qu.:0.00000             
##  Median :-0.3000       Median :0.4000         Median :0.06591             
##  Mean   :-0.2943       Mean   :0.3217         Mean   :0.17070             
##  3rd Qu.:-0.2211       3rd Qu.:0.5000         3rd Qu.:0.30000             
##  Max.   : 0.0000       Max.   :0.5000         Max.   :1.00000             
##      shares      
##  Min.   :    47  
##  1st Qu.:   832  
##  Median :  1200  
##  Mean   :  2938  
##  3rd Qu.:  2100  
##  Max.   :109100
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
{"columns":[{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"2.09297","2":"530.8388","3":"11.68493","4":"6.49638","5":"0.81295","6":"995.4392","7":"0.3431","8":"0.11207","9":"0.10095","10":"0.01265","11":"0.12875","12":"0.19257","13":"0.23036","14":"6735.624"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
{"columns":[{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"2","2":"548","3":"7","4":"1","5":"0.3241145","6":"983.4573","7":"0","8":"0.1008001","9":"0.1273887","10":"0.01431606","11":"0.1481598","12":"0.3772727","13":"0.3","14":"1268"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"1.000","2":"0.032","3":"0.019","4":"0.068","5":"-0.056","6":"0.038","7":"-0.027","8":"-0.031","9":"-0.072","10":"0.019","11":"-0.013","12":"-0.148","13":"0.041","14":"0.015","_rn_":"n_tokens_title"},{"1":"0.032","2":"1.000","3":"0.455","4":"0.242","5":"0.161","6":"-0.043","7":"0.045","8":"0.125","9":"-0.013","10":"0.176","11":"-0.155","12":"0.004","13":"-0.003","14":"-0.010","_rn_":"n_tokens_content"},{"1":"0.019","2":"0.455","3":"1.000","4":"-0.095","5":"0.048","6":"0.059","7":"0.051","8":"-0.047","9":"-0.057","10":"-0.001","11":"-0.018","12":"-0.022","13":"0.028","14":"0.045","_rn_":"num_imgs"},{"1":"0.068","2":"0.242","3":"-0.095","4":"1.000","5":"0.043","6":"0.037","7":"-0.004","8":"0.038","9":"-0.068","10":"0.242","11":"-0.117","12":"0.007","13":"0.029","14":"-0.002","_rn_":"num_videos"},{"1":"-0.056","2":"0.161","3":"0.048","4":"0.043","5":"1.000","6":"-0.114","7":"-0.008","8":"0.632","9":"0.187","10":"0.206","11":"-0.353","12":"0.061","13":"-0.049","14":"0.016","_rn_":"average_token_length"},{"1":"0.038","2":"-0.043","3":"0.059","4":"0.037","5":"-0.114","6":"1.000","7":"0.034","8":"-0.023","9":"-0.019","10":"-0.042","11":"0.000","12":"-0.018","13":"0.040","14":"0.131","_rn_":"kw_avg_avg"},{"1":"-0.027","2":"0.045","3":"0.051","4":"-0.004","5":"-0.008","6":"0.034","7":"1.000","8":"-0.007","9":"-0.003","10":"-0.007","11":"0.000","12":"-0.019","13":"-0.001","14":"0.048","_rn_":"is_weekend"},{"1":"-0.031","2":"0.125","3":"-0.047","4":"0.038","5":"0.632","6":"-0.023","7":"-0.007","8":"1.000","9":"0.231","10":"0.261","11":"-0.481","12":"0.042","13":"0.038","14":"0.056","_rn_":"global_subjectivity"},{"1":"-0.072","2":"-0.013","3":"-0.057","4":"-0.068","5":"0.187","6":"-0.019","7":"-0.003","8":"0.231","9":"1.000","10":"-0.549","11":"0.282","12":"0.026","13":"0.000","14":"0.018","_rn_":"global_sentiment_polarity"},{"1":"0.019","2":"0.176","3":"-0.001","4":"0.242","5":"0.206","6":"-0.042","7":"-0.007","8":"0.261","9":"-0.549","10":"1.000","11":"-0.393","12":"-0.064","13":"0.067","14":"-0.008","_rn_":"global_rate_negative_words"},{"1":"-0.013","2":"-0.155","3":"-0.018","4":"-0.117","5":"-0.353","6":"0.000","7":"0.000","8":"-0.481","9":"0.282","10":"-0.393","11":"1.000","12":"-0.020","13":"-0.082","14":"-0.031","_rn_":"avg_negative_polarity"},{"1":"-0.148","2":"0.004","3":"-0.022","4":"0.007","5":"0.061","6":"-0.018","7":"-0.019","8":"0.042","9":"0.026","10":"-0.064","11":"-0.020","12":"1.000","13":"-0.376","14":"0.001","_rn_":"abs_title_subjectivity"},{"1":"0.041","2":"-0.003","3":"0.028","4":"0.029","5":"-0.049","6":"0.040","7":"-0.001","8":"0.038","9":"0.000","10":"0.067","11":"-0.082","12":"-0.376","13":"1.000","14":"0.022","_rn_":"abs_title_sentiment_polarity"},{"1":"0.015","2":"-0.010","3":"0.045","4":"-0.002","5":"0.016","6":"0.131","7":"0.048","8":"0.056","9":"0.018","10":"-0.008","11":"-0.031","12":"0.001","13":"0.022","14":"1.000","_rn_":"shares"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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

![](entertainment_files/figure-html/boxplot-outliers-1.png)<!-- -->

```r
boxplot(training_data$shares,horizontal = TRUE, range = 2, outline = FALSE,main = "Boxplot of shares without outliers")
```

![](entertainment_files/figure-html/boxplot-outliers-2.png)<!-- -->

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

![](entertainment_files/figure-html/shares-vs-keywords-average-1.png)<!-- -->

We can measure the trend of shares as a function of Average number of key words. If we see a possitive trend we can say that the more key words in the articles the more likely it is to be shared, the opposite can also be said. We measure the correlation to get a more precise gauge in case the graph is not clear enough.

```r
correlation2 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$average_token_length)

plot2 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = average_token_length)) +
geom_density_2d() + 
  labs(title = "number of shares vs. Average length of words in content", y= "# of shares", x = "Average length of words in content") +
  geom_text(color = "red",x=5,y=3500,label = paste0("Correlation = ",round(correlation2,3)))

plot2
```

![](entertainment_files/figure-html/shares-vs-average-length-of-words-in-content-1.png)<!-- -->

With a density plot as a function of average length of words in content we see where most of our shares come from. We can utilize this to help explain our model down below.


```r
correlation3 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$n_tokens_content)

plot3 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = n_tokens_content)) +
geom_rug() +
  labs(title = "number of shares vs. number of words in content", y= "# of shares", x = "# of words in content") +
  geom_text(color = "red",x=4000,y=4000,label = paste0("Correlation = ",round(correlation3,3)))

plot3
```

![](entertainment_files/figure-html/density-plot-1.png)<!-- -->

Using a rug graph we can measure the relationship between number of words in content and the number of shares. The intersection between where both rugs are highly concentrated is where how we can measure correlation. If both rugs are concentrated near zero than we see that the less words the more shareable the articles are or vice versa.


```r
correlation4 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$n_tokens_title)

plot4 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = n_tokens_title)) +
geom_col() +
  labs(title = "number of shares vs. number of words in title", y= "# of shares", x = "# of words in title") +
  geom_text(color = "red",x=15,y=600000,label = paste0("Correlation = ",round(correlation4,3)))

plot4
```

![](entertainment_files/figure-html/rug-graph-1.png)<!-- -->
We see how the `# of words in title` as distributed with respect to number of shares. Any large skewness would be a flag for us to research further.


Here we graphically depict the correlations among the variables in the training data.

```r
# Note need to probably name graphs still etc...work in progress
corrplot(cor(training_data), tl.col = "black")
```

![](entertainment_files/figure-html/correlation-plot-1.png)<!-- -->

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

![](entertainment_files/figure-html/sentiment-correlation-1.png)<!-- -->

```
## 
## [[2]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](entertainment_files/figure-html/sentiment-correlation-2.png)<!-- -->

```
## 
## [[3]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](entertainment_files/figure-html/sentiment-correlation-3.png)<!-- -->

```
## 
## [[4]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](entertainment_files/figure-html/sentiment-correlation-4.png)<!-- -->

```
## 
## [[5]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](entertainment_files/figure-html/sentiment-correlation-5.png)<!-- -->

```
## 
## [[6]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](entertainment_files/figure-html/sentiment-correlation-6.png)<!-- -->


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

![](entertainment_files/figure-html/shares-weekend-plot-1.png)<!-- -->

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
##  -9816  -2063  -1451   -491 103733 
## 
## Coefficients:
##                                                                   Estimate
## (Intercept)                                                      1.961e+04
## kw_avg_avg                                                      -4.569e+00
## average_token_length                                            -5.034e+03
## n_tokens_content                                                -6.721e+01
## n_tokens_title                                                  -1.861e+03
## kw_avg_avg:average_token_length                                  1.358e+00
## kw_avg_avg:n_tokens_content                                      2.461e-02
## average_token_length:n_tokens_content                            1.493e+01
## kw_avg_avg:n_tokens_title                                        4.953e-01
## average_token_length:n_tokens_title                              4.853e+02
## n_tokens_content:n_tokens_title                                  7.452e+00
## kw_avg_avg:average_token_length:n_tokens_content                -5.370e-03
## kw_avg_avg:average_token_length:n_tokens_title                  -1.252e-01
## kw_avg_avg:n_tokens_content:n_tokens_title                      -2.750e-03
## average_token_length:n_tokens_content:n_tokens_title            -1.664e+00
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title  6.042e-04
##                                                                 Std. Error
## (Intercept)                                                      1.400e+04
## kw_avg_avg                                                       3.637e+00
## average_token_length                                             3.085e+03
## n_tokens_content                                                 4.685e+01
## n_tokens_title                                                   1.242e+03
## kw_avg_avg:average_token_length                                  8.056e-01
## kw_avg_avg:n_tokens_content                                      1.485e-02
## average_token_length:n_tokens_content                            1.018e+01
## kw_avg_avg:n_tokens_title                                        3.222e-01
## average_token_length:n_tokens_title                              2.744e+02
## n_tokens_content:n_tokens_title                                  4.297e+00
## kw_avg_avg:average_token_length:n_tokens_content                 3.208e-03
## kw_avg_avg:average_token_length:n_tokens_title                   7.163e-02
## kw_avg_avg:n_tokens_content:n_tokens_title                       1.368e-03
## average_token_length:n_tokens_content:n_tokens_title             9.373e-01
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title  2.971e-04
##                                                                 t value
## (Intercept)                                                       1.401
## kw_avg_avg                                                       -1.256
## average_token_length                                             -1.632
## n_tokens_content                                                 -1.435
## n_tokens_title                                                   -1.498
## kw_avg_avg:average_token_length                                   1.686
## kw_avg_avg:n_tokens_content                                       1.658
## average_token_length:n_tokens_content                             1.466
## kw_avg_avg:n_tokens_title                                         1.537
## average_token_length:n_tokens_title                               1.769
## n_tokens_content:n_tokens_title                                   1.734
## kw_avg_avg:average_token_length:n_tokens_content                 -1.674
## kw_avg_avg:average_token_length:n_tokens_title                   -1.748
## kw_avg_avg:n_tokens_content:n_tokens_title                       -2.010
## average_token_length:n_tokens_content:n_tokens_title             -1.776
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title   2.034
##                                                                 Pr(>|t|)  
## (Intercept)                                                       0.1614  
## kw_avg_avg                                                        0.2090  
## average_token_length                                              0.1028  
## n_tokens_content                                                  0.1514  
## n_tokens_title                                                    0.1341  
## kw_avg_avg:average_token_length                                   0.0918 .
## kw_avg_avg:n_tokens_content                                       0.0975 .
## average_token_length:n_tokens_content                             0.1428  
## kw_avg_avg:n_tokens_title                                         0.1243  
## average_token_length:n_tokens_title                               0.0770 .
## n_tokens_content:n_tokens_title                                   0.0829 .
## kw_avg_avg:average_token_length:n_tokens_content                  0.0942 .
## kw_avg_avg:average_token_length:n_tokens_title                    0.0805 .
## kw_avg_avg:n_tokens_content:n_tokens_title                        0.0445 *
## average_token_length:n_tokens_content:n_tokens_title              0.0758 .
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title   0.0420 *
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 6674 on 4923 degrees of freedom
## Multiple R-squared:  0.02117,	Adjusted R-squared:  0.01819 
## F-statistic:   7.1 on 15 and 4923 DF,  p-value: 1.174e-15
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
## -10985  -2074  -1383   -407 103475 
## 
## Coefficients:
##                                               Estimate Std. Error t value
## (Intercept)                                 -9.875e+02  8.410e+02  -1.174
## n_tokens_content                            -5.332e+00  2.925e+00  -1.823
## num_imgs                                    -6.350e+01  3.782e+01  -1.679
## num_videos                                   1.621e+02  2.509e+02   0.646
## kw_avg_avg                                   7.673e-01  1.517e-01   5.059
## global_sentiment_polarity                   -1.939e+03  1.915e+03  -1.012
## global_rate_negative_words                  -1.743e+04  1.670e+04  -1.044
## abs_title_sentiment_polarity                 2.756e+02  5.268e+02   0.523
## average_token_length                        -1.144e+02  1.915e+02  -0.597
## global_subjectivity                          5.286e+03  1.312e+03   4.028
## n_tokens_content:num_imgs                   -9.729e-03  7.403e-03  -1.314
## n_tokens_content:num_videos                 -1.666e-02  2.285e-02  -0.729
## n_tokens_content:average_token_length        1.008e+00  5.984e-01   1.685
## n_tokens_content:kw_avg_avg                 -2.312e-04  2.524e-04  -0.916
## n_tokens_content:global_sentiment_polarity   6.364e+00  3.183e+00   1.999
## n_tokens_content:global_rate_negative_words  2.690e+01  2.579e+01   1.043
## num_imgs:kw_avg_avg                          3.255e-02  1.070e-02   3.041
## num_imgs:abs_title_sentiment_polarity        6.014e+00  3.593e+01   0.167
## num_videos:average_token_length             -5.243e+00  5.160e+01  -0.102
## num_videos:global_subjectivity              -3.795e+01  2.859e+02  -0.133
## num_videos:global_sentiment_polarity        -5.196e+02  3.359e+02  -1.547
## num_videos:global_rate_negative_words       -1.831e+03  1.769e+03  -1.035
## num_videos:abs_title_sentiment_polarity     -4.171e+00  6.242e+01  -0.067
##                                             Pr(>|t|)    
## (Intercept)                                  0.24038    
## n_tokens_content                             0.06840 .  
## num_imgs                                     0.09320 .  
## num_videos                                   0.51834    
## kw_avg_avg                                  4.37e-07 ***
## global_sentiment_polarity                    0.31136    
## global_rate_negative_words                   0.29659    
## abs_title_sentiment_polarity                 0.60087    
## average_token_length                         0.55022    
## global_subjectivity                         5.70e-05 ***
## n_tokens_content:num_imgs                    0.18880    
## n_tokens_content:num_videos                  0.46606    
## n_tokens_content:average_token_length        0.09213 .  
## n_tokens_content:kw_avg_avg                  0.35974    
## n_tokens_content:global_sentiment_polarity   0.04561 *  
## n_tokens_content:global_rate_negative_words  0.29700    
## num_imgs:kw_avg_avg                          0.00237 ** 
## num_imgs:abs_title_sentiment_polarity        0.86707    
## num_videos:average_token_length              0.91906    
## num_videos:global_subjectivity               0.89441    
## num_videos:global_sentiment_polarity         0.12194    
## num_videos:global_rate_negative_words        0.30062    
## num_videos:abs_title_sentiment_polarity      0.94672    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 6656 on 4916 degrees of freedom
## Multiple R-squared:  0.02783,	Adjusted R-squared:  0.02348 
## F-statistic: 6.396 on 22 and 4916 DF,  p-value: < 2.2e-16
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

![](entertainment_files/figure-html/fitting-tree-model-1.png)<!-- -->

We are able to use the tree function to see where our variables are most important and in what order. This could change based on subject.

We can kick off a random forest model in our to see if adding this level of complexity for our model is needed/beneficial.

```r
#Random Forest
#Train control options for ensemble models
trCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

rfFit <- train(shares ~kw_avg_avg + average_token_length + n_tokens_content + n_tokens_title, data = training_data, method = "rf",trControl=trCtrl, preProcess = c("center", "scale"),tuneGrid = data.frame(mtry = 1:4))

plot(rfFit)
```

![](entertainment_files/figure-html/random-forest-1.png)<!-- -->

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

![](entertainment_files/figure-html/boosted-tree-results-1.png)<!-- -->

```r
#Results from model training
boostTree$results
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["shrinkage"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["interaction.depth"],"name":[2],"type":["int"],"align":["right"]},{"label":["n.minobsinnode"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["n.trees"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["RMSE"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["Rsquared"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["MAE"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["RMSESD"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["RsquaredSD"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["MAESD"],"name":[10],"type":["dbl"],"align":["right"]}],"data":[{"1":"0.1","2":"1","3":"10","4":"10","5":"6654.014","6":"0.01706583","7":"2856.582","8":"746.0302","9":"0.009917861","10":"145.6305","_rn_":"1"},{"1":"0.1","2":"2","3":"10","4":"10","5":"6650.460","6":"0.01591587","7":"2853.758","8":"740.9937","9":"0.008410379","10":"142.2024","_rn_":"7"},{"1":"0.1","2":"3","3":"10","4":"10","5":"6656.088","6":"0.01439374","7":"2852.507","8":"736.5724","9":"0.007412205","10":"150.1890","_rn_":"13"},{"1":"0.1","2":"4","3":"10","4":"10","5":"6649.563","6":"0.01695308","7":"2851.597","8":"740.5599","9":"0.008719232","10":"144.5363","_rn_":"19"},{"1":"0.1","2":"1","3":"10","4":"25","5":"6640.619","6":"0.01929359","7":"2841.326","8":"741.1560","9":"0.010372008","10":"150.9076","_rn_":"2"},{"1":"0.1","2":"2","3":"10","4":"25","5":"6646.672","6":"0.01903074","7":"2843.627","8":"732.0254","9":"0.007549140","10":"149.4465","_rn_":"8"},{"1":"0.1","2":"3","3":"10","4":"25","5":"6668.247","6":"0.01632762","7":"2840.602","8":"719.7856","9":"0.008020724","10":"145.3517","_rn_":"14"},{"1":"0.1","2":"4","3":"10","4":"25","5":"6674.190","6":"0.01735081","7":"2842.809","8":"722.8561","9":"0.007923622","10":"147.8728","_rn_":"20"},{"1":"0.1","2":"1","3":"10","4":"50","5":"6639.662","6":"0.02098943","7":"2826.453","8":"730.9478","9":"0.011080527","10":"144.3404","_rn_":"3"},{"1":"0.1","2":"2","3":"10","4":"50","5":"6659.892","6":"0.02135589","7":"2841.800","8":"711.9843","9":"0.010947093","10":"155.3290","_rn_":"9"},{"1":"0.1","2":"3","3":"10","4":"50","5":"6705.658","6":"0.01575025","7":"2855.459","8":"720.8975","9":"0.008512418","10":"148.5558","_rn_":"15"},{"1":"0.1","2":"4","3":"10","4":"50","5":"6715.620","6":"0.01712645","7":"2875.139","8":"720.4700","9":"0.008171970","10":"162.3218","_rn_":"21"},{"1":"0.1","2":"1","3":"10","4":"100","5":"6642.208","6":"0.02236398","7":"2827.388","8":"720.4734","9":"0.011413884","10":"137.8233","_rn_":"4"},{"1":"0.1","2":"2","3":"10","4":"100","5":"6692.038","6":"0.02011476","7":"2863.647","8":"722.1283","9":"0.011208053","10":"157.8789","_rn_":"10"},{"1":"0.1","2":"3","3":"10","4":"100","5":"6738.775","6":"0.01660782","7":"2880.241","8":"707.1566","9":"0.009431316","10":"142.9312","_rn_":"16"},{"1":"0.1","2":"4","3":"10","4":"100","5":"6773.471","6":"0.01639861","7":"2901.384","8":"707.9271","9":"0.008698522","10":"158.0970","_rn_":"22"},{"1":"0.1","2":"1","3":"10","4":"150","5":"6640.730","6":"0.02308684","7":"2820.178","8":"727.3130","9":"0.011702068","10":"141.6509","_rn_":"5"},{"1":"0.1","2":"2","3":"10","4":"150","5":"6720.803","6":"0.01791911","7":"2871.914","8":"712.3090","9":"0.010294259","10":"159.4131","_rn_":"11"},{"1":"0.1","2":"3","3":"10","4":"150","5":"6774.326","6":"0.01489982","7":"2904.289","8":"701.6958","9":"0.007930239","10":"142.2197","_rn_":"17"},{"1":"0.1","2":"4","3":"10","4":"150","5":"6805.907","6":"0.01605077","7":"2938.099","8":"690.5145","9":"0.009864276","10":"154.5783","_rn_":"23"},{"1":"0.1","2":"1","3":"10","4":"200","5":"6644.186","6":"0.02327365","7":"2824.774","8":"721.2673","9":"0.011408951","10":"137.7075","_rn_":"6"},{"1":"0.1","2":"2","3":"10","4":"200","5":"6748.348","6":"0.01678393","7":"2887.320","8":"717.3596","9":"0.010429135","10":"168.7749","_rn_":"12"},{"1":"0.1","2":"3","3":"10","4":"200","5":"6797.525","6":"0.01502677","7":"2927.023","8":"700.0106","9":"0.008436411","10":"147.0465","_rn_":"18"},{"1":"0.1","2":"4","3":"10","4":"200","5":"6844.687","6":"0.01494879","7":"2968.668","8":"686.7058","9":"0.008496788","10":"165.8736","_rn_":"24"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["shrinkage"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["interaction.depth"],"name":[2],"type":["int"],"align":["right"]},{"label":["n.minobsinnode"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["n.trees"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["RMSE"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["Rsquared"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["MAE"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["RMSESD"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["RsquaredSD"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["MAESD"],"name":[10],"type":["dbl"],"align":["right"]}],"data":[{"1":"0.1","2":"1","3":"10","4":"10","5":"6654.014","6":"0.01706583","7":"2856.582","8":"746.0302","9":"0.009917861","10":"145.6305","_rn_":"1"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
|Linear Model 1 | 3039.060| 10073.88|
|Linear Model 2 | 3050.875| 10081.94|
|Random Forrest | 3216.057| 10492.72|
|Boosted Tree   | 3018.646| 10090.96|

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
