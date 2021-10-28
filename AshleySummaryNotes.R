#General summary
summary(subset_data)

#Removing url from data frame
subData <- subset_data %>% select(timedelta:shares)


#Checking data for NA  or infinite values
apply(subset_data, 2, function(x) any(is.na(x) | is.infinite(x)))

#Standard deviation of each variable minus url
> options(scipen = 999)
> SDs <- sapply(subData, sd)
> SDs

ggplot(subData, aes(x = global_rate_negative_words, y = shares)) + geom_point()+
  geom_smooth(method=lm, se=TRUE)

ggplot(subData, aes(x = global_rate_negative_words, y = shares)) + geom_point()+
  geom_smooth(method="auto", se=TRUE, fullrange=FALSE, level=0.95)


if abs(correlation) >= 0.5 but != 1
print pairs 


variables <- as_tibble(attributes(subData)$names) %>% rename(variable = "value") %>% bind_cols(correlations)
join with

corr <- cor(subData, method = "pearson”)
round(corr, 2)
Correlations <-as_tibble(corr, rownames = attributes(subData)$names)
)

CorrMatrix <- bind_cols(variable, corr)
column_to_rownames(CorrMatrix, var = variables)
