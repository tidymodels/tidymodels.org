library(tidyverse)
library(sf)
library(RColorBrewer)
library(ragg)

data(building_complaints, package = "modeldatatoo")

building_complaints <- building_complaints %>% 
  mutate(odd_location = if_else(borough == "Brooklyn" & latitude > 41, TRUE, FALSE)) %>% 
  filter(!odd_location) %>% 
  select(-odd_location)

# Source:
# https://mapcruzin.com/free-united-states-shapefiles/free-new-york-arcgis-maps-shapefiles.htm
# Look for "New York Highway Shapefile". This also requires "new_york_highway.shx"
ny_roads <- st_read(dsn = "learn/statistics/survival-case-study/new_york_highway.shp") 

borough_cols <- brewer.pal(5, "Dark2")
names(borough_cols) <- c("Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island")

nyc_x <- extendrange(building_complaints$longitude)
nyc_y <- extendrange(building_complaints$latitude)
nyc_ratio <- diff(nyc_x)/diff(nyc_y)

all_nyc <- 
  ggplot() +
  xlim(nyc_x) +
  ylim(nyc_y) + 
  theme_void() + 
  theme(legend.position = "bottom", legend.title = element_blank()) +
  geom_sf(data = ny_roads, aes(geometry = geometry), alpha = .1) +
  geom_point(
    data = building_complaints,
    aes(
      x = longitude,
      y = latitude,
      col = borough
    ),
    size = 1, 
    alpha = .5
  ) + 
  scale_color_manual(values = borough_cols) 

agg_png("nyc_building_complaints.png", width = 820 * 3, height = 820 * 3, res = 300, scaling = 1)
print(all_nyc)
dev.off()