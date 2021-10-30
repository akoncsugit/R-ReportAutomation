downSized <- subset_data %>% select(n_tokens_title, n_tokens_content,
                                    num_imgs:average_token_length,
                                    kw_avg_avg, is_weekend,
                                    global_subjectivity,
                                    global_sentiment_polarity,
                                    global_rate_negative_words,
                                    avg_negative_polarity,
                                    abs_title_subjectivity:shares)
corrplot(cor(downSized))
