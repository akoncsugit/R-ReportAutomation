ST558 Project 2
================
Sergio Mora & Ashley Ko

# Introduction

This project creates and compares predictions for online news popularity
data sets derived from Mashable. The goal is to predict the number of
`shares` for each of the six data channels (lifestyle, entertainment,
business, social media, tech, world). A description and summary of this
data set, `OnlineNewsPopularity.csv`, can be found here [Online News
Popularity Data
Set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).

In order to achieve this goal, we first subset data based on predictors
we hypothesize to have an impact the number of shares of per article. We
chose following predictors: `n_tokens_title`, `n_tokens_content`,
`num_imgs`, `num_videos`, `average_token_length`, `kw_avg_avg`,
`is_weekend`, `global_subjectivity`, `global_sentiment_polarity`,
`global_rate_negative_words`, `avg_negative_polarity`,
`abs_title_subjectivity`, `abs_title_sentiment_polarity`.

First, we selected average key words(kw\_avg\_avg), average length of
words (average\_token\_length), content (n\_tokens\_content), and title
(n\_tokens\_title) as metrics for the readability of an article. We
believe that easy to read articles might be more likely to be shared. In
tandem with readability, articles with more images(num\_imgs) and
videos(num\_videos) might be easier to consume and more likely shared.
It is possible that the number of shares is linked whether an article is
published on a weekday or weekend (is\_weekend). Generally speaking,
people have more more free time on the weekends which leads to more
screen time and perhaps, more articles being shared. Lastly, emotionally
charged content is often compelling so we investigated the impact of the
subjectivity(global\_subjective and abs\_title\_subjectivity), sentiment
polarity(global\_sentiment\_polarity and
abs\_title\_sentiment\_polarity), and negative word rate
(global\_rate\_negative\_words).

A variety of numerical and graphic summaries have been used to display
our chosen predictors and their relationship to the number of shares for
the training data set. We generated the following numeric summaries
minimums, 1st quartiles, medians, means, 3rd quartiles, maximums,
standard deviations, IQRs, and correlations. Additionally, we used
correlation plots, boxplots, rug graphs, and a few others to visualize
the training data. We used both linear models and ensemble methods
(random forest and boosted tree) to generate predictions.

## Libraries and Set-Up

This report uses the a `set.seed(123)` and the following libraries
`knitr`, `rmarkdown`, `caret`, `tidyvers`, `corrplot`, and `tree`.

# Data

## Reading in data

We will read in our csv dataset. We will also split our data by
`data_channel_is_*`.

``` r
# Reads in csv file `OnlineNewsPopularity/OnlineNewsPopularity.csv` 
# to produce a data frame
online_new_popularity_data <- 
  read.csv("./OnlineNewsPopularity/OnlineNewsPopularity.csv")
```

## Subsetting the data

We will subset the data based on the category listed in our YAML header.
In this case, using data from `data_channel_is_entertainment`. We will
remove any non-predictors such as `url` and `timedelta` and selected our
desired predictors\*\* and `shares`.

\*\*`n_tokens_title`, `n_tokens_content`, `num_imgs`, `num_videos`,
`average_token_length`, `kw_avg_avg`, `is_weekend`,
`global_subjectivity`, `global_sentiment_polarity`,
`global_rate_negative_words`, `avg_negative_polarity`,
`abs_title_subjectivity`, `abs_title_sentiment_polarity`

``` r
# Subsetting our data based on the category parameter, dropping non-predictors,
# and subsetting for desired predictors and response variables
subset_data <- online_new_popularity_data %>%
  filter(!!as.name(paste0("data_channel_is_",params$category)) == 1) %>%
  select(n_tokens_title, n_tokens_content, num_imgs:average_token_length,
         kw_avg_avg, is_weekend, global_subjectivity, global_sentiment_polarity, 
         global_rate_negative_words, avg_negative_polarity,
         abs_title_subjectivity, abs_title_sentiment_polarity, shares)
```

Next, we will check for potential problematic values such as NA or
infinity. These could result in errors with later analysis. Should a
problem arise later on, this allows for a diagnostic to rule out
potential problematic values.

``` r
# Checking data for NA  or infinite values
na_or_infinite <- as.data.frame(apply(subset_data,
                                2, function(x) any(is.na(x) | is.infinite(x))))
colnames(na_or_infinite) <- c("NA or Infinite values")
na_or_infinite %>% kable()
```

|                                 | NA or Infinite values |
| :------------------------------ | :-------------------- |
| n\_tokens\_title                | FALSE                 |
| n\_tokens\_content              | FALSE                 |
| num\_imgs                       | FALSE                 |
| num\_videos                     | FALSE                 |
| average\_token\_length          | FALSE                 |
| kw\_avg\_avg                    | FALSE                 |
| is\_weekend                     | FALSE                 |
| global\_subjectivity            | FALSE                 |
| global\_sentiment\_polarity     | FALSE                 |
| global\_rate\_negative\_words   | FALSE                 |
| avg\_negative\_polarity         | FALSE                 |
| abs\_title\_subjectivity        | FALSE                 |
| abs\_title\_sentiment\_polarity | FALSE                 |
| shares                          | FALSE                 |

In this chunk, we will split our newly subset data frame into training
and test data sets. We will use a simple 70/30 split

``` r
# Setting up a simple 70/30 split for our already subset data
sample_size <- floor(0.7 * nrow(subset_data))
train_ind <- sample(seq_len(nrow(subset_data)), size = sample_size)

# This will be needed later on when we start modeling
training_data <- subset_data[train_ind,]
test_data <- subset_data[-train_ind,]
```

# Summarizations

This first step in any data analysis project is exploration of the data
via numeric and graphical summaries. Here we will use common statistics
like mean, standard deviation, correlation numerically summarize the
training data. Then we will create a variety of graphical represenations
of the training data.

## Numeric Summary

### Six Number Summary

First, let’s perform a simple six number summary of all variables from
the training data set. This summary includes minimum, 1st quartile,
median, mean, 3rd quartile, and maximum values. This provides a senses
of scale and range for variable values. Note: Binary response
variable`is_weekend` has as values of 0 or 1.

``` r
#Generates a six number summary from training_data
summary <- summary(training_data)
summary
```

    ##  n_tokens_title n_tokens_content    num_imgs         num_videos     average_token_length
    ##  Min.   : 2     Min.   :   0.0   Min.   :  0.000   Min.   : 0.000   Min.   :0.000       
    ##  1st Qu.:10     1st Qu.: 254.0   1st Qu.:  1.000   1st Qu.: 0.000   1st Qu.:4.429       
    ##  Median :11     Median : 433.0   Median :  1.000   Median : 1.000   Median :4.586       
    ##  Mean   :11     Mean   : 606.4   Mean   :  6.278   Mean   : 2.636   Mean   :4.475       
    ##  3rd Qu.:12     3rd Qu.: 802.0   3rd Qu.:  8.000   3rd Qu.: 1.000   3rd Qu.:4.753       
    ##  Max.   :18     Max.   :6505.0   Max.   :128.000   Max.   :74.000   Max.   :7.696       
    ##    kw_avg_avg      is_weekend     global_subjectivity global_sentiment_polarity
    ##  Min.   :    0   Min.   :0.0000   Min.   :0.0000      Min.   :-0.37393         
    ##  1st Qu.: 2551   1st Qu.:0.0000   1st Qu.:0.4116      1st Qu.: 0.04643         
    ##  Median : 2972   Median :0.0000   Median :0.4632      Median : 0.11099         
    ##  Mean   : 3152   Mean   :0.1363   Mean   :0.4519      Mean   : 0.11082         
    ##  3rd Qu.: 3535   3rd Qu.:0.0000   3rd Qu.:0.5124      3rd Qu.: 0.17382         
    ##  Max.   :14138   Max.   :1.0000   Max.   :0.8333      Max.   : 0.72784         
    ##  global_rate_negative_words avg_negative_polarity abs_title_subjectivity
    ##  Min.   :0.00000            Min.   :-1.0000       Min.   :0.0000        
    ##  1st Qu.:0.01061            1st Qu.:-0.3692       1st Qu.:0.1227        
    ##  Median :0.01712            Median :-0.3000       Median :0.4000        
    ##  Mean   :0.01905            Mean   :-0.2943       Mean   :0.3217        
    ##  3rd Qu.:0.02492            3rd Qu.:-0.2211       3rd Qu.:0.5000        
    ##  Max.   :0.09104            Max.   : 0.0000       Max.   :0.5000        
    ##  abs_title_sentiment_polarity     shares      
    ##  Min.   :0.00000              Min.   :    47  
    ##  1st Qu.:0.00000              1st Qu.:   832  
    ##  Median :0.06591              Median :  1200  
    ##  Mean   :0.17070              Mean   :  2938  
    ##  3rd Qu.:0.30000              3rd Qu.:  2100  
    ##  Max.   :1.00000              Max.   :109100

### Standard Deviation

The previous section does not generate standard deviation for the
variable values. Standard deviation is necessary for determining the
variance of the response and predictors. It is a good diagnostic to spot
potential issues that violate assumptions necessary for models and
analysis. Here we will produce standard deviations for each variable
from `training_data.` Note: Binary response variable`is_weekend` has as
values of 0 or 1.

``` r
# Removes scientific notation for readability
options(scipen = 999)

# Generates and rounds (for simplicity) standard deviation
train_SDs <- as_tibble(lapply(training_data[, 1:14], sd))
round(train_SDs, digits = 5)

# Turns scientific notation on
options(scipen = 0)
```

### IQR

Although the 1st and 3rd quartiles are identified in the six number
summary, it is helpful quantify the range between these two values or
IQR. IQR is also needed for subsequent plotting. Note: Binary response
variable`is_weekend` has as values of 0 or 1.

``` r
# Uses lapply to generate IQR for all variables in training_data
IQR<- as_tibble(lapply(training_data[, 1:14], IQR))
IQR
```

### Correlations

Prior to preforming any model fitting or statistical analysis, it is
essential to understand the potential correlation among predictors and
between the response and predictors. Correlation helps identify
potential collinearity and, thus, allows for better candidate model
selection. It is worth noting any absolute correlation values \> 0.5.
However, this threshold has been left to discretion of the individual.

``` r
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

## Graphical summaries

We suppose that part of what makes a link shareable is how easy it is
for the content to be consumed. Smaller, bite sized information, in a
simple writing style might be easier to process and result in more
shares. We will test this out via proxy’s. We will measure shares
against average key words(kw\_avg\_avg), average length of words
(average\_token\_length), average number of words in the content
(n\_tokens\_content), and number of words in the title
(n\_tokens\_title). The idea here is to measure both the quantity of
words as well as the complexity of the content. i.e. an article with 500
“easy” words could be shared more than an article with 100 “difficult”
words.

### Improving graphical summaries - Cutlier check

To produce graphs and plots that give an accurate sense of the data, we
will search for potential. If we have any outliers we will remove them
first to get an idea of what the bulk of shares come from. We will
follow what the boxplot tells us when choosing what to remove.

``` r
# Boxplot from training_data
boxplot(training_data$shares,horizontal = TRUE, range = 2,
        main = "Boxplot of shares with outliers")
```

![](entertainment_files/figure-gfm/boxplot-outliers-1.png)<!-- -->

``` r
boxplot(training_data$shares,horizontal = TRUE, range = 2, outline = FALSE,
        main = "Boxplot of shares without outliers")
```

![](entertainment_files/figure-gfm/boxplot-outliers-2.png)<!-- -->

``` r
# Creates a subset of training data without outliers for #plotting purposes
IQR <- quantile(training_data$shares)[4] - quantile(subset_data$shares)[2]
upper_limit <- quantile(training_data$shares)[4] + (1.5 * IQR)
lower_limit <- quantile(training_data$shares)[2] - (1.5 * IQR)
subset_data_wo_outliers <- training_data %>%
  filter(shares <= upper_limit & shares >= lower_limit)
```

After we remove any potential outliers to our data our we can compare
shares our key metrics.

``` r
correlation1 <- cor(subset_data_wo_outliers$shares,
                    subset_data_wo_outliers$kw_avg_avg)

plot1 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = kw_avg_avg)) + 
  geom_point() +
  geom_smooth() +
  labs(title = "Number of shares vs. Average number of key words",
       y= "# of shares", x = "Average # of key words") +
  geom_text(color = "red",x=15000,y=5000,
            label = paste0("Correlation = ",round(correlation1,3)))

plot1
```

    ## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'

![](entertainment_files/figure-gfm/shares-vs-keywords-average-1.png)<!-- -->

We can measure the trend of shares as a function of Average number of
key words. If we see a positive trend we can say that the more key words
in the articles the more likely it is to be shared, the opposite can
also be said. We measure the correlation to get a more precise gauge in
case the graph is not clear enough.

``` r
correlation2 <- cor(subset_data_wo_outliers$shares,
                    subset_data_wo_outliers$average_token_length)

plot2 <- ggplot(subset_data_wo_outliers, 
                aes(y= shares,x = average_token_length)) +
  geom_density_2d() + 
  labs(title = "number of shares vs. Average length of words in content",
       y= "# of shares", x = "Average length of words in content") +
  geom_text(color = "red",x=5,y=3500,label = paste0("Correlation = ",
                                                    round(correlation2,3)))

plot2
```

![](entertainment_files/figure-gfm/shares-vs-average-length-of-words-in-content-1.png)<!-- -->

With a density plot as a function of average length of words in content
we see where most of our shares come from. We can utilize this to help
explain our model down below.

``` r
correlation3 <- cor(subset_data_wo_outliers$shares,
                    subset_data_wo_outliers$n_tokens_content)

plot3 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = n_tokens_content)) +
geom_rug() +
  labs(title = "number of shares vs. number of words in content",
       y= "# of shares", x = "# of words in content") +
  geom_text(color = "red",x=4000,
            y=4000,label = paste0("Correlation = ",round(correlation3,3)))

plot3
```

![](entertainment_files/figure-gfm/density-plot-1.png)<!-- -->

Using a rug graph we can measure the relationship between number of
words in content and the number of shares. The intersection between
where both rugs are highly concentrated is where how we can measure
correlation. If both rugs are concentrated near zero than we see that
the less words the more shareable the articles are or vice versa.

``` r
correlation4 <- cor(subset_data_wo_outliers$shares,
                    subset_data_wo_outliers$n_tokens_title)

plot4 <- ggplot(subset_data_wo_outliers, aes(y= shares,x = n_tokens_title)) +
geom_col() +
  labs(title = "number of shares vs. number of words in title",
       y= "# of shares", x = "# of words in title") +
  geom_text(color = "red",x=15,y=600000,label = paste0("Correlation = ",
                                                       round(correlation4,3)))

plot4
```

![](entertainment_files/figure-gfm/rug-graph-1.png)<!-- --> We see how
the `# of words in title` as distributed with respect to number of
shares. Any large skewness would be a flag for us to research further.

Here we graphically depict the correlations among the variables in the
training data.

``` r
# Note need to probably name graphs still etc...work in progress
corrplot(cor(training_data), tl.col = "black")
```

![](entertainment_files/figure-gfm/correlation-plot-1.png)<!-- -->

In addition to the previous correlation plot, it is necessary to look at
individual scatterplots of the shares vs. predictors. Previously, we
plotted the relationship between shares and various word counts, but in
this section we will focus exclusively on the emotion predictors such as
`global_subjectivity`, `global_sentiment_polarity`,
`global_rate_negative_words`, `avg_negative_polarity`,
`abs_title_subjectivity`, and `abs_title_sentiment_polarity`.

``` r
corNoOut<- function(x) {
  var1 <- get(x, subset_data_wo_outliers)
  title <-paste0("Scatterplot of shares vs ", x)
  xlabel <- x
  g <- ggplot(subset_data_wo_outliers, aes(x = var1, y = shares)) +
    geom_point() +
    geom_smooth() + geom_smooth(method = lm, col = "Red") +
    labs(title = title, y= "# of shares", x = xlabel)
}
sentiment_preds <- list("global_subjectivity", "global_sentiment_polarity",
                        "global_rate_negative_words", "avg_negative_polarity",
                        "abs_title_subjectivity",
                        "abs_title_sentiment_polarity")
lapply(sentiment_preds, corNoOut)
```

    ## [[1]]

    ## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'

    ## `geom_smooth()` using formula 'y ~ x'

![](entertainment_files/figure-gfm/sentiment-correlation-1.png)<!-- -->

    ## 
    ## [[2]]

    ## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
    ## `geom_smooth()` using formula 'y ~ x'

![](entertainment_files/figure-gfm/sentiment-correlation-2.png)<!-- -->

    ## 
    ## [[3]]

    ## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
    ## `geom_smooth()` using formula 'y ~ x'

![](entertainment_files/figure-gfm/sentiment-correlation-3.png)<!-- -->

    ## 
    ## [[4]]

    ## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
    ## `geom_smooth()` using formula 'y ~ x'

![](entertainment_files/figure-gfm/sentiment-correlation-4.png)<!-- -->

    ## 
    ## [[5]]

    ## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
    ## `geom_smooth()` using formula 'y ~ x'

![](entertainment_files/figure-gfm/sentiment-correlation-5.png)<!-- -->

    ## 
    ## [[6]]

    ## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
    ## `geom_smooth()` using formula 'y ~ x'

![](entertainment_files/figure-gfm/sentiment-correlation-6.png)<!-- -->

We also believe that shares might increase based on whether the post was
created on a weekend. Perhaps, weekend posts are shared more frequently
as, generally, people have more screen time and thus are more apt to
share article. This section `shares` has been scaled and density plotted
by `is_weekend` as a factor where 0 is a weekday and 1 is a weekend day.

``` r
# Removing scientific notation for readability
options(scipen = 999)

# Coverts is_weekend into a two level factor
weekend_factor <- subset_data_wo_outliers %>%
  mutate(weekend = as.factor(is_weekend)) %>% select(weekend, shares)

# Base plot
g <- ggplot(weekend_factor, aes(x=shares)) +xlab("Number of Shares") +
  ylab("Density")

# Filled, stacked histogram with density of shares by weekend level
g + geom_histogram(aes(y = ..density.., fill = weekend)) + 
  geom_density(adjust = 0.25, alpha = 0.5, aes(fill = weekend),
               position = "stack") +
  labs(title = "Density of Shares: Weekday vs. Weekend") + 
  scale_fill_discrete(name = "Weekday or Weekend?",
                      labels = c("Weekday", "Weekend"))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](entertainment_files/figure-gfm/shares-weekend-plot-1.png)<!-- -->

``` r
# Turning on scientific notation
options(scipen = 0)
```

# Modeling

## Linear Models

Linear regression is the process of modeling a response through the best
fit line with predictors. Models are fit by minimizing the sum of
squared residuals. We can also determine the success of a linear model
by R^2 values. Here are some common examples of the equations which
generate a prediction for the response y with predictor x and error term
epsilon.

Linear model with one predictor x.
\[y = \beta_0 + \beta_1x_1 + \epsilon\] Main effects only model with i
predictors.
\[y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ...+ \beta_ix_i + \epsilon\]
Main effects and additive terms with 2 predictors
\[y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_3x_1x_2 + \epsilon\]
Fitting our linear model with our chosen variables. In this first linear
model we will investigate our suposition that readability of the title
and content increase shares.

``` r
# First linear model
lmfit1 <- 
  lm(shares ~kw_avg_avg*average_token_length*n_tokens_content*n_tokens_title,
     data = training_data)
summary(lmfit1)
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
    ##                                                                   Estimate Std. Error
    ## (Intercept)                                                      1.961e+04  1.400e+04
    ## kw_avg_avg                                                      -4.569e+00  3.637e+00
    ## average_token_length                                            -5.034e+03  3.085e+03
    ## n_tokens_content                                                -6.721e+01  4.685e+01
    ## n_tokens_title                                                  -1.861e+03  1.242e+03
    ## kw_avg_avg:average_token_length                                  1.358e+00  8.056e-01
    ## kw_avg_avg:n_tokens_content                                      2.461e-02  1.485e-02
    ## average_token_length:n_tokens_content                            1.493e+01  1.018e+01
    ## kw_avg_avg:n_tokens_title                                        4.953e-01  3.222e-01
    ## average_token_length:n_tokens_title                              4.853e+02  2.744e+02
    ## n_tokens_content:n_tokens_title                                  7.452e+00  4.297e+00
    ## kw_avg_avg:average_token_length:n_tokens_content                -5.370e-03  3.208e-03
    ## kw_avg_avg:average_token_length:n_tokens_title                  -1.252e-01  7.163e-02
    ## kw_avg_avg:n_tokens_content:n_tokens_title                      -2.750e-03  1.368e-03
    ## average_token_length:n_tokens_content:n_tokens_title            -1.664e+00  9.373e-01
    ## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title  6.042e-04  2.971e-04
    ##                                                                 t value Pr(>|t|)  
    ## (Intercept)                                                       1.401   0.1614  
    ## kw_avg_avg                                                       -1.256   0.2090  
    ## average_token_length                                             -1.632   0.1028  
    ## n_tokens_content                                                 -1.435   0.1514  
    ## n_tokens_title                                                   -1.498   0.1341  
    ## kw_avg_avg:average_token_length                                   1.686   0.0918 .
    ## kw_avg_avg:n_tokens_content                                       1.658   0.0975 .
    ## average_token_length:n_tokens_content                             1.466   0.1428  
    ## kw_avg_avg:n_tokens_title                                         1.537   0.1243  
    ## average_token_length:n_tokens_title                               1.769   0.0770 .
    ## n_tokens_content:n_tokens_title                                   1.734   0.0829 .
    ## kw_avg_avg:average_token_length:n_tokens_content                 -1.674   0.0942 .
    ## kw_avg_avg:average_token_length:n_tokens_title                   -1.748   0.0805 .
    ## kw_avg_avg:n_tokens_content:n_tokens_title                       -2.010   0.0445 *
    ## average_token_length:n_tokens_content:n_tokens_title             -1.776   0.0758 .
    ## kw_avg_avg:average_token_length:n_tokens_content:n_tokens_title   2.034   0.0420 *
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6674 on 4923 degrees of freedom
    ## Multiple R-squared:  0.02117,    Adjusted R-squared:  0.01819 
    ## F-statistic:   7.1 on 15 and 4923 DF,  p-value: 1.174e-15

Using the `data_source_is_lifestyle` as base data source, the following
chunk of code was run and to determine potential significant additive
and interaction terms by t tests for significance with p-values less
than or equal to 0.05. The subsequent model was selected for this report
is `lm`(`shares` \~ `n_tokens_content*num_imgs` +
`n_tokens_content*num_videos` + `n_tokens_content:average_token_length`
+ `n_tokens_content*kw_avg_avg` +
`n_tokens_content*global_sentiment_polarity` +
`n_tokens_content*global_rate_negative_words` + `num_imgs*kw_avg_avg` +
`num_imgs*abs_title_sentiment_polarity` +
`num_videos*average_token_length` + `num_videos*global_subjectivity` +
`num_videos*global_sentiment_polarity` +
`num_videos*global_rate_negative_words` +
`num_videos*abs_title_sentiment_polarity`, `data = training_data`)

``` r
# Generates a linear model with all main effects and interactions terms
lm_full <- lm(shares ~ .^2, data = training_data)
summary(lm_full)
```

This is the second linear model we will use as described above it was
selected by examining p-values from the summary of the a linear model
with all main effects and interaction terms.

``` r
# Generalates second linear model candidate
lmfit2 <- lm(shares ~  n_tokens_content*num_imgs +
               n_tokens_content*num_videos + 
               n_tokens_content:average_token_length +
               n_tokens_content*kw_avg_avg + 
               n_tokens_content*global_sentiment_polarity + 
               n_tokens_content*global_rate_negative_words +
               num_imgs*kw_avg_avg + num_imgs*abs_title_sentiment_polarity +
               num_videos*average_token_length + 
               num_videos*global_subjectivity +
               num_videos*global_sentiment_polarity + 
               num_videos*global_rate_negative_words +
               num_videos*abs_title_sentiment_polarity,
              data = training_data)
summary(lmfit2)
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
    ##                                               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 -9.875e+02  8.410e+02  -1.174  0.24038    
    ## n_tokens_content                            -5.332e+00  2.925e+00  -1.823  0.06840 .  
    ## num_imgs                                    -6.350e+01  3.782e+01  -1.679  0.09320 .  
    ## num_videos                                   1.621e+02  2.509e+02   0.646  0.51834    
    ## kw_avg_avg                                   7.673e-01  1.517e-01   5.059 4.37e-07 ***
    ## global_sentiment_polarity                   -1.939e+03  1.915e+03  -1.012  0.31136    
    ## global_rate_negative_words                  -1.743e+04  1.670e+04  -1.044  0.29659    
    ## abs_title_sentiment_polarity                 2.756e+02  5.268e+02   0.523  0.60087    
    ## average_token_length                        -1.144e+02  1.915e+02  -0.597  0.55022    
    ## global_subjectivity                          5.286e+03  1.312e+03   4.028 5.70e-05 ***
    ## n_tokens_content:num_imgs                   -9.729e-03  7.403e-03  -1.314  0.18880    
    ## n_tokens_content:num_videos                 -1.666e-02  2.285e-02  -0.729  0.46606    
    ## n_tokens_content:average_token_length        1.008e+00  5.984e-01   1.685  0.09213 .  
    ## n_tokens_content:kw_avg_avg                 -2.312e-04  2.524e-04  -0.916  0.35974    
    ## n_tokens_content:global_sentiment_polarity   6.364e+00  3.183e+00   1.999  0.04561 *  
    ## n_tokens_content:global_rate_negative_words  2.690e+01  2.579e+01   1.043  0.29700    
    ## num_imgs:kw_avg_avg                          3.255e-02  1.070e-02   3.041  0.00237 ** 
    ## num_imgs:abs_title_sentiment_polarity        6.014e+00  3.593e+01   0.167  0.86707    
    ## num_videos:average_token_length             -5.243e+00  5.160e+01  -0.102  0.91906    
    ## num_videos:global_subjectivity              -3.795e+01  2.859e+02  -0.133  0.89441    
    ## num_videos:global_sentiment_polarity        -5.196e+02  3.359e+02  -1.547  0.12194    
    ## num_videos:global_rate_negative_words       -1.831e+03  1.769e+03  -1.035  0.30062    
    ## num_videos:abs_title_sentiment_polarity     -4.171e+00  6.242e+01  -0.067  0.94672    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6656 on 4916 degrees of freedom
    ## Multiple R-squared:  0.02783,    Adjusted R-squared:  0.02348 
    ## F-statistic: 6.396 on 22 and 4916 DF,  p-value: < 2.2e-16

With a simple linear model we can test it’s goodness of fit with
\(R^2\). Since we are measuring human behavior we can be comfortable
with a low \(R^2\). However too low (although subjective) would indicate
that our hypothesis is wrong. As a rule of thumb we will say:

\[R^2 < .1 : we \space suspect \space that \space we \space cannot \space reject \space H_0 \\
  R^2 \space between \space .1 \space and \space .5 \space : \space we \space suspect \space that \space we \space would \space reject \space H_0 \space \\
  R^2 > .5 \space we \space feel \space confident \space that \space our \space variables \space are \space good \space predictors \space and \space our \space hypothesis \space is \space a \space good \space explanation.\]

## Ensemble Methods

We will be using non-linear tree based methods to generate predictions
for the number of `shares` an article will receive. Tree based methods
create predictions by dividing predictors space into regions and produce
different predictions for each region. This split is made by trying to
minimize the Residual Sum of Squares. Each subsequent split is then
further split until the tree has many nodes. As a larger tree is prone
to over fitting, we used cross-validation to prune back our trees. As
our response `shares` is continuous we will be using regression trees
which uses mean of observation as a prediction for each region.

Random forest models use bootstrapping to create multiple trees and
average over the results. Predictors are randomly selected for each tree
and all predictors are used at once. To compare random forest models we
look at Root MSE values and select the model with the lowest RMSE.
Boosting tree models follow a similar procedure as trees grow
sequentially but where each tree is built to correct the errors from the
first model as for each split, the residuals are treated as the
response. The prediction is updated and the process repeated. Boosting
models are generally better but slower at prediction than random forest
models.

### Random Forest

We can now fit our model above into a tree function. This will give us a
better picture of where our variables are most important in our model.

``` r
fitTree <- tree(shares ~kw_avg_avg + average_token_length + n_tokens_content + 
                  n_tokens_title, data = training_data)
plot(fitTree)
text(fitTree)
```

![](entertainment_files/figure-gfm/fitting-tree-model-1.png)<!-- -->

We are able to use the tree function to see where our variables are most
important and in what order. This could change based on subject.

We can kick off a random forest model in our to see if adding this level
of complexity for our model is needed/beneficial.

``` r
#Random Forest
#Train control options for ensemble models
trCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

rfFit <- train(shares ~kw_avg_avg + average_token_length + n_tokens_content +
                 n_tokens_title, data = training_data, method = "rf",
               trControl=trCtrl, preProcess = c("center", "scale"),
               tuneGrid = data.frame(mtry = 1:4))

plot(rfFit)
```

![](entertainment_files/figure-gfm/random-forest-1.png)<!-- -->

By plotting the `rfFit` we can see which `mtry` value is the best. This
might be different between subjects.

### Boosted Tree

Lastly, a boosted tree was fit using 5-fold, 3 times repeated
cross-validation with tuning parameter combinations of `n.trees` = (10,
25, 50, 100, 150, and 200), `interaction.depth` = 1:4, `shrinkage` =
0.1, and `n.minobsinnode` = 10.

``` r
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

``` r
#Plot of RMSE by Max Tree Depth
plot(boostTree)
```

![](entertainment_files/figure-gfm/boosted-tree-results-1.png)<!-- -->

``` r
#Results from model training
boostTree$results
```

The best tuning paramters are:

``` r
#Best tuning parameters
boostTree$bestTune
```

This section generates the predictions based on the best boosted tree
model.

``` r
#Uses best tuned training boosted tree model to predict test data
boostTreePred <-predict(boostTree, newdata = test_data)

#Reports best boosted tree model and corresponding RMSE
boost <-boostTree$results[1,]
boost
```

# Comparison

``` r
lmfit_1 = c(MAE(test_data$shares,predict(lmfit1)),
            RMSE(test_data$shares,predict(lmfit1)))

lmfit_2 = c(MAE(test_data$shares,predict(lmfit2)),
            RMSE(test_data$shares,predict(lmfit2)))

rffit_c = c(MAE(test_data$shares,predict(rfFit)),
            RMSE(test_data$shares,predict(rfFit)))

boostTree_c = c(MAE(test_data$shares,predict(boostTree)),
                RMSE(test_data$shares,predict(boostTree)))

MAE_RMSE_SUMM <- rbind.data.frame("Linear Model 1" = lmfit_1,
                                  "Linear Model 2" = lmfit_2,
                                  "Random Forrest" = rffit_c,
                                  "Boosted Tree" = boostTree_c)

colnames(MAE_RMSE_SUMM) <- c("MAE","RMSE")
rownames(MAE_RMSE_SUMM) <- c("Linear Model 1", "Linear Model 2",
                             "Random Forrest", "Boosted Tree")
kable(MAE_RMSE_SUMM, caption = "Comparing models via MAE and RMSE")
```

|                |      MAE |     RMSE |
| :------------- | -------: | -------: |
| Linear Model 1 | 3039.060 | 10073.88 |
| Linear Model 2 | 3050.875 | 10081.94 |
| Random Forrest | 3216.057 | 10492.72 |
| Boosted Tree   | 3018.646 | 10090.96 |

Comparing models via MAE and RMSE

Measuring success with MAE and RMSE. This helps measure the models
success while also accounting for the complexity of the model. We can
use the MAE **and** RMSE to see if any issues in our data come form a
single point (or a small subset of points).

# Automation

The following code chunk was called in the console in order to generate
a report of each category of data\_channel\_is\_\* (`lifestyle`,
`entertainment`, `bus`, `socmed`, `tech`, `world`).

``` r
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
