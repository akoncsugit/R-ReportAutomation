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
We will subset the data based on the category listed in our YAML header. In this case, using data from `data_channel_is_socmed`. We will remove non-predictors such as `url` and `timedelta` and selected our desired predictors** and `shares`.
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
##  n_tokens_title   n_tokens_content    num_imgs        num_videos    
##  Min.   : 4.000   Min.   :   0.0   Min.   : 0.000   Min.   : 0.000  
##  1st Qu.: 8.000   1st Qu.: 253.0   1st Qu.: 1.000   1st Qu.: 0.000  
##  Median : 9.000   Median : 431.0   Median : 1.000   Median : 0.000  
##  Mean   : 9.615   Mean   : 609.0   Mean   : 4.104   Mean   : 1.049  
##  3rd Qu.:11.000   3rd Qu.: 760.5   3rd Qu.: 2.000   3rd Qu.: 1.000  
##  Max.   :18.000   Max.   :3735.0   Max.   :62.000   Max.   :73.000  
##  average_token_length   kw_avg_avg      is_weekend     global_subjectivity
##  Min.   :0.000        Min.   :    0   Min.   :0.0000   Min.   :0.0000     
##  1st Qu.:4.488        1st Qu.: 2635   1st Qu.:0.0000   1st Qu.:0.4061     
##  Median :4.649        Median : 3159   Median :0.0000   Median :0.4589     
##  Mean   :4.627        Mean   : 3198   Mean   :0.1261   Mean   :0.4569     
##  3rd Qu.:4.797        3rd Qu.: 3605   3rd Qu.:0.0000   3rd Qu.:0.5117     
##  Max.   :5.635        Max.   :33953   Max.   :1.0000   Max.   :0.9222     
##  global_sentiment_polarity global_rate_negative_words
##  Min.   :-0.37500          Min.   :0.000000          
##  1st Qu.: 0.08897          1st Qu.:0.009217          
##  Median : 0.14122          Median :0.014436          
##  Mean   : 0.14418          Mean   :0.015757          
##  3rd Qu.: 0.19325          3rd Qu.:0.020742          
##  Max.   : 0.56667          Max.   :0.139831          
##  avg_negative_polarity abs_title_subjectivity abs_title_sentiment_polarity
##  Min.   :-1.0000       Min.   :0.0000         Min.   :0.0000              
##  1st Qu.:-0.3167       1st Qu.:0.2000         1st Qu.:0.0000              
##  Median :-0.2492       Median :0.5000         Median :0.0000              
##  Mean   :-0.2563       Mean   :0.3545         Mean   :0.1479              
##  3rd Qu.:-0.1806       3rd Qu.:0.5000         3rd Qu.:0.2093              
##  Max.   : 0.0000       Max.   :0.5000         Max.   :1.0000              
##      shares     
##  Min.   :   23  
##  1st Qu.: 1400  
##  Median : 2200  
##  Mean   : 3654  
##  3rd Qu.: 3800  
##  Max.   :59000
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
{"columns":[{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"2.10333","2":"547.9519","3":"8.08103","4":"3.4454","5":"0.42438","6":"1240.297","7":"0.33204","8":"0.09369","9":"0.09255","10":"0.01001","11":"0.12164","12":"0.18327","13":"0.22803","14":"5242.535"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
{"columns":[{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"3","2":"507.5","3":"1","4":"1","5":"0.309335","6":"969.4213","7":"0","8":"0.105625","9":"0.1042787","10":"0.01152492","11":"0.1360417","12":"0.3","13":"0.2093209","14":"2400"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["n_tokens_title"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["n_tokens_content"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["num_imgs"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["num_videos"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["average_token_length"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["kw_avg_avg"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["is_weekend"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["global_subjectivity"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["global_sentiment_polarity"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["global_rate_negative_words"],"name":[10],"type":["dbl"],"align":["right"]},{"label":["avg_negative_polarity"],"name":[11],"type":["dbl"],"align":["right"]},{"label":["abs_title_subjectivity"],"name":[12],"type":["dbl"],"align":["right"]},{"label":["abs_title_sentiment_polarity"],"name":[13],"type":["dbl"],"align":["right"]},{"label":["shares"],"name":[14],"type":["dbl"],"align":["right"]}],"data":[{"1":"1.000","2":"-0.009","3":"0.003","4":"-0.030","5":"-0.040","6":"0.014","7":"0.043","8":"-0.043","9":"-0.026","10":"-0.001","11":"-0.031","12":"-0.109","13":"0.017","14":"-0.017","_rn_":"n_tokens_title"},{"1":"-0.009","2":"1.000","3":"0.520","4":"0.008","5":"0.056","6":"0.003","7":"0.076","8":"0.120","9":"0.006","10":"0.126","11":"-0.103","12":"0.030","13":"0.036","14":"0.034","_rn_":"n_tokens_content"},{"1":"0.003","2":"0.520","3":"1.000","4":"-0.102","5":"0.055","6":"0.031","7":"0.025","8":"0.129","9":"0.068","10":"0.062","11":"-0.020","12":"-0.028","13":"0.064","14":"-0.042","_rn_":"num_imgs"},{"1":"-0.030","2":"0.008","3":"-0.102","4":"1.000","5":"-0.030","6":"0.041","7":"-0.037","8":"0.139","9":"0.132","10":"0.017","11":"-0.028","12":"-0.002","13":"0.061","14":"0.007","_rn_":"num_videos"},{"1":"-0.040","2":"0.056","3":"0.055","4":"-0.030","5":"1.000","6":"-0.026","7":"-0.013","8":"0.247","9":"0.095","10":"0.063","11":"-0.121","12":"0.034","13":"-0.008","14":"-0.033","_rn_":"average_token_length"},{"1":"0.014","2":"0.003","3":"0.031","4":"0.041","5":"-0.026","6":"1.000","7":"0.033","8":"0.058","9":"0.026","10":"0.045","11":"-0.018","12":"0.004","13":"0.047","14":"0.097","_rn_":"kw_avg_avg"},{"1":"0.043","2":"0.076","3":"0.025","4":"-0.037","5":"-0.013","6":"0.033","7":"1.000","8":"-0.004","9":"-0.020","10":"0.032","11":"-0.081","12":"-0.070","13":"0.010","14":"0.039","_rn_":"is_weekend"},{"1":"-0.043","2":"0.120","3":"0.129","4":"0.139","5":"0.247","6":"0.058","7":"-0.004","8":"1.000","9":"0.329","10":"0.206","11":"-0.268","12":"-0.005","13":"0.140","14":"0.015","_rn_":"global_subjectivity"},{"1":"-0.026","2":"0.006","3":"0.068","4":"0.132","5":"0.095","6":"0.026","7":"-0.020","8":"0.329","9":"1.000","10":"-0.483","11":"0.313","12":"-0.029","13":"0.109","14":"-0.045","_rn_":"global_sentiment_polarity"},{"1":"-0.001","2":"0.126","3":"0.062","4":"0.017","5":"0.063","6":"0.045","7":"0.032","8":"0.206","9":"-0.483","10":"1.000","11":"-0.301","12":"-0.034","13":"0.040","14":"0.048","_rn_":"global_rate_negative_words"},{"1":"-0.031","2":"-0.103","3":"-0.020","4":"-0.028","5":"-0.121","6":"-0.018","7":"-0.081","8":"-0.268","9":"0.313","10":"-0.301","11":"1.000","12":"-0.023","13":"-0.040","14":"-0.073","_rn_":"avg_negative_polarity"},{"1":"-0.109","2":"0.030","3":"-0.028","4":"-0.002","5":"0.034","6":"0.004","7":"-0.070","8":"-0.005","9":"-0.029","10":"-0.034","11":"-0.023","12":"1.000","13":"-0.422","14":"0.020","_rn_":"abs_title_subjectivity"},{"1":"0.017","2":"0.036","3":"0.064","4":"0.061","5":"-0.008","6":"0.047","7":"0.010","8":"0.140","9":"0.109","10":"0.040","11":"-0.040","12":"-0.422","13":"1.000","14":"0.055","_rn_":"abs_title_sentiment_polarity"},{"1":"-0.017","2":"0.034","3":"-0.042","4":"0.007","5":"-0.033","6":"0.097","7":"0.039","8":"0.015","9":"-0.045","10":"0.048","11":"-0.073","12":"0.020","13":"0.055","14":"1.000","_rn_":"shares"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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

![](socmed_files/figure-html/boxplot-outliers-1.png)<!-- -->

```r
boxplot(training_data$shares,horizontal = TRUE, range = 2, outline = FALSE,main = "Boxplot of shares without outliers")
```

![](socmed_files/figure-html/boxplot-outliers-2.png)<!-- -->

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

![](socmed_files/figure-html/shares-vs-keywords-average-1.png)<!-- -->

We can measure the trend of shares as a function of Average number of key words. If we see a possitive trend we can say that the more key words in the articles the more likely it is to be shared, the opposite can also be said. We measure the correlation to get a more precise gauge in case the graph is not clear enough.

```r
correlation2 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$average_token_length)

plot2 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = average_token_length)) +
geom_density_2d() + 
  labs(title = "number of shares vs. Average length of words in content", y= "# of shares", x = "Average length of words in content") +
  geom_text(color = "red",x=5,y=3500,label = paste0("Correlation = ",round(correlation2,3)))

plot2
```

![](socmed_files/figure-html/shares-vs-average-length-of-words-in-content-1.png)<!-- -->

With a density plot as a function of average length of words in content we see where most of our shares come from. We can utilize this to help explain our model down below.


```r
correlation3 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$n_tokens_content)

plot3 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = n_tokens_content)) +
geom_rug() +
  labs(title = "number of shares vs. number of words in content", y= "# of shares", x = "# of words in content") +
  geom_text(color = "red",x=4000,y=4000,label = paste0("Correlation = ",round(correlation3,3)))

plot3
```

![](socmed_files/figure-html/density-plot-1.png)<!-- -->

Using a rug graph we can measure the relationship between number of words in content and the number of shares. The intersection between where both rugs are highly concentrated is where how we can measure correlation. If both rugs are concentrated near zero than we see that the less words the more shareable the articles are or vice versa.


```r
correlation4 <- cor(subset_data_wo_outliers$shares,subset_data_wo_outliers$n_tokens_title)

plot4 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = n_tokens_title)) +
geom_col() +
  labs(title = "number of shares vs. number of words in title", y= "# of shares", x = "# of words in title") +
  geom_text(color = "red",x=15,y=600000,label = paste0("Correlation = ",round(correlation4,3)))

plot4
```

![](socmed_files/figure-html/rug-graph-1.png)<!-- -->
We see how the `# of words in title` as distributed with respect to number of shares. Any large skewness would be a flag for us to research further.


Here we graphically depict the correlations among the variables in the training data.

```r
# Note need to probably name graphs still etc...work in progress
corrplot(cor(training_data), tl.col = "black")
```

![](socmed_files/figure-html/correlation-plot-1.png)<!-- -->

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

![](socmed_files/figure-html/sentiment-correlation-1.png)<!-- -->

```
## 
## [[2]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](socmed_files/figure-html/sentiment-correlation-2.png)<!-- -->

```
## 
## [[3]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](socmed_files/figure-html/sentiment-correlation-3.png)<!-- -->

```
## 
## [[4]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](socmed_files/figure-html/sentiment-correlation-4.png)<!-- -->

```
## 
## [[5]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](socmed_files/figure-html/sentiment-correlation-5.png)<!-- -->

```
## 
## [[6]]
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
## `geom_smooth()` using formula 'y ~ x'
```

![](socmed_files/figure-html/sentiment-correlation-6.png)<!-- -->


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

![](socmed_files/figure-html/shares-weekend-plot-1.png)<!-- -->

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
## -12239  -2151  -1339    271  55195 
## 
## Coefficients:
##                                                                   Estimate
## (Intercept)                                                     -1.938e+04
## kw_avg_avg                                                       3.062e+00
## average_token_length                                             4.751e+03
## n_tokens_content                                                 9.423e+01
## n_tokens_title                                                   1.362e+03
## kw_avg_avg:average_token_length                                 -6.310e-01
## kw_avg_avg:n_tokens_content                                     -2.397e-02
## average_token_length:n_tokens_content                           -2.020e+01
## kw_avg_avg:n_tokens_title                                       -2.913e-02
## average_token_length:n_tokens_title                             -2.907e+02
## n_tokens_content:n_tokens_title                                 -8.154e+00
## kw_avg_avg:average_token_length:n_tokens_content                 5.274e-03
## kw_avg_avg:average_token_length:n_tokens_title                   8.102e-03
## kw_avg_avg:n_tokens_content:n_tokens_title                       2.458e-03
## average_token_length:n_tokens_content:n_tokens_title             1.730e+00
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title -5.340e-04
##                                                                 Std. Error
## (Intercept)                                                      4.437e+04
## kw_avg_avg                                                       1.248e+01
## average_token_length                                             9.557e+03
## n_tokens_content                                                 9.485e+01
## n_tokens_title                                                   3.771e+03
## kw_avg_avg:average_token_length                                  2.683e+00
## kw_avg_avg:n_tokens_content                                      2.688e-02
## average_token_length:n_tokens_content                            2.054e+01
## kw_avg_avg:n_tokens_title                                        1.031e+00
## average_token_length:n_tokens_title                              8.149e+02
## n_tokens_content:n_tokens_title                                  9.054e+00
## kw_avg_avg:average_token_length:n_tokens_content                 5.831e-03
## kw_avg_avg:average_token_length:n_tokens_title                   2.225e-01
## kw_avg_avg:n_tokens_content:n_tokens_title                       2.572e-03
## average_token_length:n_tokens_content:n_tokens_title             1.965e+00
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title  5.591e-04
##                                                                 t value
## (Intercept)                                                      -0.437
## kw_avg_avg                                                        0.245
## average_token_length                                              0.497
## n_tokens_content                                                  0.993
## n_tokens_title                                                    0.361
## kw_avg_avg:average_token_length                                  -0.235
## kw_avg_avg:n_tokens_content                                      -0.892
## average_token_length:n_tokens_content                            -0.983
## kw_avg_avg:n_tokens_title                                        -0.028
## average_token_length:n_tokens_title                              -0.357
## n_tokens_content:n_tokens_title                                  -0.901
## kw_avg_avg:average_token_length:n_tokens_content                  0.905
## kw_avg_avg:average_token_length:n_tokens_title                    0.036
## kw_avg_avg:n_tokens_content:n_tokens_title                        0.956
## average_token_length:n_tokens_content:n_tokens_title              0.880
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title  -0.955
##                                                                 Pr(>|t|)
## (Intercept)                                                        0.662
## kw_avg_avg                                                         0.806
## average_token_length                                               0.619
## n_tokens_content                                                   0.321
## n_tokens_title                                                     0.718
## kw_avg_avg:average_token_length                                    0.814
## kw_avg_avg:n_tokens_content                                        0.373
## average_token_length:n_tokens_content                              0.326
## kw_avg_avg:n_tokens_title                                          0.977
## average_token_length:n_tokens_title                                0.721
## n_tokens_content:n_tokens_title                                    0.368
## kw_avg_avg:average_token_length:n_tokens_content                   0.366
## kw_avg_avg:average_token_length:n_tokens_title                     0.971
## kw_avg_avg:n_tokens_content:n_tokens_title                         0.339
## average_token_length:n_tokens_content:n_tokens_title               0.379
## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title    0.340
## 
## Residual standard error: 5198 on 1610 degrees of freedom
## Multiple R-squared:  0.02594,	Adjusted R-squared:  0.01687 
## F-statistic: 2.859 on 15 and 1610 DF,  p-value: 0.0001889
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
## -11983  -2119  -1309    401  55812 
## 
## Coefficients:
##                                               Estimate Std. Error t value
## (Intercept)                                  3.161e+03  1.756e+03   1.800
## n_tokens_content                             1.147e+01  4.689e+00   2.446
## num_imgs                                    -2.086e+01  6.660e+01  -0.313
## num_videos                                  -2.595e+03  1.059e+03  -2.451
## kw_avg_avg                                   2.401e-01  1.986e-01   1.209
## global_sentiment_polarity                   -1.165e+03  2.599e+03  -0.448
## global_rate_negative_words                   3.791e+03  2.298e+04   0.165
## abs_title_sentiment_polarity                 1.391e+03  6.928e+02   2.007
## average_token_length                        -2.133e+02  3.777e+02  -0.565
## global_subjectivity                          3.097e+02  1.847e+03   0.168
## n_tokens_content:num_imgs                   -4.582e-02  2.245e-02  -2.041
## n_tokens_content:num_videos                 -2.657e-02  4.066e-02  -0.654
## n_tokens_content:average_token_length       -2.184e+00  9.587e-01  -2.279
## n_tokens_content:kw_avg_avg                  1.961e-04  3.387e-04   0.579
## n_tokens_content:global_sentiment_polarity  -5.743e+00  4.035e+00  -1.423
## n_tokens_content:global_rate_negative_words -8.517e+00  3.572e+01  -0.238
## num_imgs:kw_avg_avg                          1.818e-02  1.700e-02   1.069
## num_imgs:abs_title_sentiment_polarity       -4.652e+01  6.299e+01  -0.739
## num_videos:average_token_length              4.551e+02  2.164e+02   2.103
## num_videos:global_subjectivity               3.838e+02  5.196e+02   0.739
## num_videos:global_sentiment_polarity         7.989e+02  5.280e+02   1.513
## num_videos:global_rate_negative_words        9.565e+03  6.176e+03   1.549
## num_videos:abs_title_sentiment_polarity      5.443e+01  1.691e+02   0.322
##                                             Pr(>|t|)  
## (Intercept)                                   0.0720 .
## n_tokens_content                              0.0146 *
## num_imgs                                      0.7541  
## num_videos                                    0.0144 *
## kw_avg_avg                                    0.2268  
## global_sentiment_polarity                     0.6539  
## global_rate_negative_words                    0.8690  
## abs_title_sentiment_polarity                  0.0449 *
## average_token_length                          0.5724  
## global_subjectivity                           0.8669  
## n_tokens_content:num_imgs                     0.0414 *
## n_tokens_content:num_videos                   0.5135  
## n_tokens_content:average_token_length         0.0228 *
## n_tokens_content:kw_avg_avg                   0.5626  
## n_tokens_content:global_sentiment_polarity    0.1549  
## n_tokens_content:global_rate_negative_words   0.8116  
## num_imgs:kw_avg_avg                           0.2850  
## num_imgs:abs_title_sentiment_polarity         0.4603  
## num_videos:average_token_length               0.0356 *
## num_videos:global_subjectivity                0.4602  
## num_videos:global_sentiment_polarity          0.1304  
## num_videos:global_rate_negative_words         0.1216  
## num_videos:abs_title_sentiment_polarity       0.7475  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 5167 on 1603 degrees of freedom
## Multiple R-squared:  0.04159,	Adjusted R-squared:  0.02843 
## F-statistic: 3.162 on 22 and 1603 DF,  p-value: 1.131e-06
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

![](socmed_files/figure-html/fitting-tree-model-1.png)<!-- -->

We are able to use the tree function to see where our variables are most important and in what order. This could change based on subject.

We can kick off a random forest model in our to see if adding this level of complexity for our model is needed/beneficial.

```r
#Random Forest
#Train control options for ensemble models
trCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

rfFit <- train(shares ~kw_avg_avg + average_token_length + n_tokens_content + n_tokens_title, data = training_data, method = "rf",trControl=trCtrl, preProcess = c("center", "scale"),tuneGrid = data.frame(mtry = 1:4))

plot(rfFit)
```

![](socmed_files/figure-html/random-forest-1.png)<!-- -->

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

![](socmed_files/figure-html/boosted-tree-results-1.png)<!-- -->

```r
#Results from model training
boostTree$results
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["shrinkage"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["interaction.depth"],"name":[2],"type":["int"],"align":["right"]},{"label":["n.minobsinnode"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["n.trees"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["RMSE"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["Rsquared"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["MAE"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["RMSESD"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["RsquaredSD"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["MAESD"],"name":[10],"type":["dbl"],"align":["right"]}],"data":[{"1":"0.1","2":"1","3":"10","4":"10","5":"5101.362","6":"0.018433919","7":"2662.955","8":"1094.9170","9":"0.012748336","10":"255.3187","_rn_":"1"},{"1":"0.1","2":"2","3":"10","4":"10","5":"5106.857","6":"0.015356566","7":"2674.601","8":"1084.1749","9":"0.012739579","10":"242.6344","_rn_":"7"},{"1":"0.1","2":"3","3":"10","4":"10","5":"5111.725","6":"0.016892794","7":"2675.623","8":"1093.9684","9":"0.018781742","10":"243.8121","_rn_":"13"},{"1":"0.1","2":"4","3":"10","4":"10","5":"5118.421","6":"0.014513228","7":"2689.150","8":"1083.5324","9":"0.011126533","10":"243.6228","_rn_":"19"},{"1":"0.1","2":"1","3":"10","4":"25","5":"5108.806","6":"0.015157863","7":"2671.880","8":"1085.2398","9":"0.010115504","10":"263.7861","_rn_":"2"},{"1":"0.1","2":"2","3":"10","4":"25","5":"5141.734","6":"0.013425116","7":"2706.621","8":"1058.0034","9":"0.011733356","10":"229.7676","_rn_":"8"},{"1":"0.1","2":"3","3":"10","4":"25","5":"5155.952","6":"0.014242789","7":"2716.672","8":"1075.6773","9":"0.012673706","10":"253.5537","_rn_":"14"},{"1":"0.1","2":"4","3":"10","4":"25","5":"5172.011","6":"0.014447407","7":"2713.929","8":"1051.0561","9":"0.012895623","10":"232.1156","_rn_":"20"},{"1":"0.1","2":"1","3":"10","4":"50","5":"5121.938","6":"0.016991574","7":"2683.799","8":"1073.9781","9":"0.010742255","10":"254.6328","_rn_":"3"},{"1":"0.1","2":"2","3":"10","4":"50","5":"5168.489","6":"0.015764458","7":"2710.844","8":"1050.7715","9":"0.016306961","10":"231.2780","_rn_":"9"},{"1":"0.1","2":"3","3":"10","4":"50","5":"5222.266","6":"0.013873879","7":"2762.868","8":"1048.0856","9":"0.011867708","10":"234.7651","_rn_":"15"},{"1":"0.1","2":"4","3":"10","4":"50","5":"5242.773","6":"0.013286018","7":"2763.699","8":"1052.0128","9":"0.014022905","10":"250.6821","_rn_":"21"},{"1":"0.1","2":"1","3":"10","4":"100","5":"5145.165","6":"0.017885589","7":"2706.124","8":"1057.0189","9":"0.011579213","10":"230.8802","_rn_":"4"},{"1":"0.1","2":"2","3":"10","4":"100","5":"5221.022","6":"0.015119647","7":"2752.676","8":"1039.4985","9":"0.015871587","10":"236.4137","_rn_":"10"},{"1":"0.1","2":"3","3":"10","4":"100","5":"5291.352","6":"0.011713227","7":"2810.237","8":"1037.7740","9":"0.010875116","10":"239.7314","_rn_":"16"},{"1":"0.1","2":"4","3":"10","4":"100","5":"5332.571","6":"0.010406682","7":"2851.785","8":"1013.3516","9":"0.010886351","10":"243.0756","_rn_":"22"},{"1":"0.1","2":"1","3":"10","4":"150","5":"5150.514","6":"0.018765654","7":"2721.668","8":"1058.1831","9":"0.011805091","10":"232.7879","_rn_":"5"},{"1":"0.1","2":"2","3":"10","4":"150","5":"5272.853","6":"0.013223268","7":"2793.840","8":"1017.4163","9":"0.017672056","10":"221.2244","_rn_":"11"},{"1":"0.1","2":"3","3":"10","4":"150","5":"5357.040","6":"0.009790002","7":"2863.403","8":"1006.5597","9":"0.008980239","10":"235.6107","_rn_":"17"},{"1":"0.1","2":"4","3":"10","4":"150","5":"5419.640","6":"0.008163735","7":"2919.287","8":"1002.2249","9":"0.007502329","10":"229.8880","_rn_":"23"},{"1":"0.1","2":"1","3":"10","4":"200","5":"5148.609","6":"0.020195929","7":"2720.096","8":"1053.4067","9":"0.013124763","10":"238.7830","_rn_":"6"},{"1":"0.1","2":"2","3":"10","4":"200","5":"5308.849","6":"0.011812345","7":"2817.997","8":"996.7905","9":"0.014027906","10":"205.1708","_rn_":"12"},{"1":"0.1","2":"3","3":"10","4":"200","5":"5421.024","6":"0.009477283","7":"2927.216","8":"977.3103","9":"0.009544270","10":"213.4945","_rn_":"18"},{"1":"0.1","2":"4","3":"10","4":"200","5":"5485.417","6":"0.007318113","7":"2987.123","8":"982.0875","9":"0.008562126","10":"214.6386","_rn_":"24"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["shrinkage"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["interaction.depth"],"name":[2],"type":["int"],"align":["right"]},{"label":["n.minobsinnode"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["n.trees"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["RMSE"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["Rsquared"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["MAE"],"name":[7],"type":["dbl"],"align":["right"]},{"label":["RMSESD"],"name":[8],"type":["dbl"],"align":["right"]},{"label":["RsquaredSD"],"name":[9],"type":["dbl"],"align":["right"]},{"label":["MAESD"],"name":[10],"type":["dbl"],"align":["right"]}],"data":[{"1":"0.1","2":"1","3":"10","4":"10","5":"5101.362","6":"0.01843392","7":"2662.955","8":"1094.917","9":"0.01274834","10":"255.3187","_rn_":"1"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
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
|Linear Model 1 | 2743.932| 6051.473|
|Linear Model 2 | 2783.039| 6063.227|
|Random Forrest | 2968.676| 6506.921|
|Boosted Tree   | 2711.242| 5999.157|

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
