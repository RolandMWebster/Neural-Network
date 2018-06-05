# Visualising our Data ----------------------------------------------------

# Stack our data
train.tidied <- train.data %>%
  head(9) %>%
  rename("value" = V1) %>%
  mutate(observation = row_number()) %>%
  gather(key = "pixel", value = "pixel_value",-value,-observation) %>%
  mutate(pixel = as.numeric(gsub("V","",pixel))-1) %>%
  arrange(observation, pixel) %>%
  mutate(x_pos = ((pixel - 1) %% kImageWidth + 1),
         y_pos = abs(29-(((pixel - 1) %/% kImageWidth) + 1)),
         pixel_value = pixel_value / max(pixel_value)) %>%
  dplyr::select(-pixel)

# Plot
ggplot(train.tidied,
       aes(x = x_pos,
           y = y_pos,
           fill = pixel_value)) +
  geom_tile() +
  geom_text(aes(x = 0, y = 0, label = value, col = "red")) +
  facet_wrap(~observation)
