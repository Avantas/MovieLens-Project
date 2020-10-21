##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################
# like the first step, we will create the train and validation sets
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

dim(edx) # the dimensions of the dataset

head(edx) 

edx %>% filter(rating == 3) %>% nrow() # the number of califications with 3

#q3
#How many different movies are in the edx dataset?
edx %>% group_by(movieId) %>% summarize(mean = mean(rating)) #number of diferents movies

#q4
#How many different users are in the edx dataset?
edx %>% group_by(userId) %>% summarize(mean = mean(rating)) %>% nrow() #number of diferents users

set.seed(1, sample.kind="Rounding")

# is not neccesary now,.. but
pattern <- "^\\d*Drama\\d*$" 
#q5
#How many movie ratings are in each of the following genres in the edx dataset?
str_subset(edx$genres, "Drama")
str_subset(edx$genres, "Comedy")
str_subset(edx$genres, "Thriller")
str_subset(edx$genres, "Romance")

#q6
#Which movie has the greatest number of ratings?

str_subset(edx$title, "Forrest Gump") #we check and change the name in c
str_subset(edx$title, "Jurassic Park")
str_subset(edx$title, "Pulp Fiction")
str_subset(edx$title, "Shawshank Redemption")
str_subset(edx$title, "Speed 2")

#I change the names

str_subset(edx$title, "Forrest Gump (\\d*)")
str_subset(edx$title, "Jurassic Park, (\\d*)") # agrego la coma para diferenciar las versiones de peliculas y restar
str_subset(edx$title, "Pulp Fiction (\\d*)")   # the movie with more rows
str_subset(edx$title, "Shawshank Redemption, The (\\d*)")
str_subset(edx$title, "Speed 2: Cruise Control (\\d*)")

#q7

edx %>% group_by(rating) %>% ggplot(aes(rating)) + geom_density()

####### PROJECT ########

# At the start the project (with the data previus generate), we have two objets with will work,
#It is the edx (where the information we train and test), and validation object (with the test set
#to test when we finish out analysis)
# The fisrt step, is to take edx object and make a data partition:
set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, 
                                  list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]



# for be sure there are the same users and movies in the test set and training set;

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#we investigate the train_set

train_set %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

train_set$genres

# for compute the RMSE we make the funtion, so we will mesure the eficience between our prediction and the true validation ratings
#So, we can see how we make better our prediction.
# Root mean squared error RMSE

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}                             

# The simplest model

mu_hat <- mean(train_set$rating)
mu_hat

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

#For sure we can do this better, we save the results in a new object to compare leter.

rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)
rmse_results

# We will modeling the "movie effect" for count or discount to our first total mean mu_hat

mu <- mean(train_set$rating) 

movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_effect <- RMSE(predicted_ratings, test_set$rating)
movie_effect

#We also will model the user effect to add to mu

train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

user_effect <- RMSE(predicted_ratings, test_set$rating)
user_effect

# we will regularizate for penalize the large estimates using small samples size for predict better

# for cross validation we take 5
# all the model togheter

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

lambda_opt <- lambdas[which.min(rmses)]
lambda_opt

regularizated_movie_user <- min(rmses)

#for compare

rmse_results <- tibble(method = c("Just the average", "Movie effect Model", "Movie + User effect Model", "Regularized Movie + User effect Model"), RMSE = c(naive_rmse, movie_effect, user_effect, regularizated_movie_user))
rmse_results

#year of the movie

head(train_set$title)

year_movie <-as.numeric(str_sub(train_set$title, start = -5, end = -2))
head(year_movie)
train_set <- train_set %>% mutate(year = year_movie)

year_movie <-as.numeric(str_sub(test_set$title, start = -5, end = -2))
test_set <- test_set %>% mutate(year = year_movie)

year_movie <-as.numeric(str_sub(validation$title, start = -5, end = -2))
validation <- validation %>% mutate(year = year_movie)

year_movie <-as.numeric(str_sub(edx$title, start = -5, end = -2))
edx <- edx %>% mutate(year = year_movie)

# we add a column with the age of the movie

validation <- validation %>% mutate(Age = 2020 - year)
train_set <- train_set %>% mutate(Age = 2020 - year)
test_set <- test_set %>% mutate(Age = 2020 - year)
edx <- edx %>% mutate(Age = 2020 - year)


head(train_set)

# also we need to convert the timestamp in a year of rating
class(validation$timestamp)

validation <- validation %>% mutate(timestamp = as.Date.POSIXct(timestamp))
validation <- validation %>% mutate(year_rating = year(timestamp))
head(validation)

train_set <- train_set %>% mutate(timestamp = as.Date.POSIXct(timestamp))
train_set <- train_set %>% mutate(year_rating = year(timestamp))

test_set <- test_set %>% mutate(timestamp = as.Date.POSIXct(timestamp))
test_set <- test_set %>% mutate(year_rating = year(timestamp))

edx <- edx %>% mutate(timestamp = as.Date.POSIXct(timestamp))
edx <- edx %>% mutate(year_rating = year(timestamp))

# for gendres,

genres <- train_set %>% separate_rows(genres, sep = "\\|")

table_genres <- genres %>% group_by(genres) %>% summarize(n = n(), mean = mean(rating)) %>% arrange(desc(n))
table_genres %>% mutate(genres = reorder(genres, mean)) %>% ggplot(aes(genres,mean,size = n, color = n)) + geom_count() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + ggtitle("Genres Vs Means and rating") + xlab("Genres") + ylab("Means")
table_genres %>% mutate(dif = mean - mean(mean))


#we add the age effect

mu <- mean(train_set$rating)
  
b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+4.75))
  
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+4.75))

b_age <- train_set %>%
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by ="userId") %>%
  group_by(Age) %>% 
  summarize(b_age = mean(rating - b_i - b_u - mu)/(n()+4.75))
  
predicted_ratings <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_age, by = "Age") %>%
  mutate(pred = mu + b_i + b_u + b_age) %>%
  pull(pred)
  


regu_user_movie_age <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- tibble(method = c("Just the average", "Movie effect Model", "Movie + User effect Model", "Regularized Movie + User effect Model", "Regularized Movie + User + Age effect Model"), RMSE = c(naive_rmse, movie_effect, user_effect, regularizated_movie_user, regu_user_movie_age)) %>% mutate(RMSE = sprintf("%0.5f", RMSE))
rmse_results


#its not represent a lot of change
#now we move the MU

mu <- mean(train_set$rating)+seq(-0.2, 0.7, 0.05)


rmses <- sapply(mu, function(mu){
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+4.75))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+4.75))
  b_age <- train_set %>%
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by ="userId") %>%
    group_by(Age) %>% 
    summarize(b_age = mean(rating - b_i - b_u - mu)/(n()+4.75))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_age, by = "Age") %>%
    mutate(pred = mu + b_i + b_u + b_age) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(mu, rmses)  

mu <- mu[which.min(rmses)]
mu
regu_user_movie_age_mu <- min(rmses)

rmse_results <- tibble(method = c("Just the average", "Movie effect Model", "Movie + User effect Model", "Regularized Movie + User effect Model", "Regularized Movie + User + Age effect Model", "Reg. Movie + User + Age effect Model (best mu)"), RMSE = c(naive_rmse, movie_effect, user_effect, regularizated_movie_user, regu_user_movie_age, regu_user_movie_age_mu)) %>% mutate(RMSE = sprintf("%0.5f", RMSE))
rmse_results

regu_user_movie_age_mu

#another option is model the gendre
#we can see, 797 diferents conbinations for gendre. We will model it.

train_set %>% group_by(genres) %>% summarize(b_gen = mu-mean(rating))

#


mu <- mean(train_set$rating)

b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+4.75))

b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+4.75))

b_age <- train_set %>%
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by ="userId") %>%
  group_by(Age) %>% 
  summarize(b_age = sum(rating - b_i - b_u - mu)/(n()+4.75))

b_gen <- train_set %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by ="userId") %>%
  left_join(b_age, by = "Age") %>%
  group_by(genres) %>% 
  summarize(b_gen = sum(rating - b_i - b_u - b_age - mu)/(n()+4.75))

predicted_ratings <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_age, by = "Age") %>%
  left_join(b_gen, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_age) %>%
  pull(pred)

regu_user_movie_age_gen <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- tibble(method = c("Just the average", "Movie effect Model", "Movie + User effect Model", "Regularized Movie + User effect Model", "Regularized Movie + User + Age effect Model", "Reg. Movie + User + Age + Genre effect Model"), RMSE = c(naive_rmse, movie_effect, user_effect, regularizated_movie_user, regu_user_movie_age, regu_user_movie_age_gen)) %>% mutate(RMSE = sprintf("%0.5f", RMSE))
rmse_results

#we ajust again the mu 



mu <- mean(train_set$rating)+seq(-0.2, 0.7, 0.05)


rmses <- sapply(mu, function(mu){
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+4.75))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+4.75))
  
  b_age <- train_set %>%
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by ="userId") %>%
    group_by(Age) %>% 
    summarize(b_age = sum(rating - b_i - b_u - mu)/(n()+4.75))
  
  b_gen <- train_set %>% 
    left_join(b_i, by="movieId") %>% 
    left_join(b_u, by ="userId") %>%
    left_join(b_age, by = "Age") %>%
    group_by(genres) %>% 
    summarize(b_gen = sum(rating - b_i - b_u - b_age - mu)/(n()+4.75))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_age, by = "Age") %>%
    left_join(b_gen, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_age) %>%
    pull(pred)

  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(mu, rmses)  

mu[which.min(rmses)]

regu_user_movie_age_gen_mu <- min(rmses)

rmse_results <- tibble(method = c("Just the average", "Movie effect Model", "Movie + User effect Model", "Regularized Movie + User effect Model", "Regularized Movie + User + Age effect Model", "Reg. Movie + User + Age + Genre effect Model", "Reg. Movie + User + Age + Genre effect Model (mu opt)"), RMSE = c(naive_rmse, movie_effect, user_effect, regularizated_movie_user, regu_user_movie_age, regu_user_movie_age_gen, regu_user_movie_age_gen_mu)) %>% mutate(RMSE = sprintf("%0.5f", RMSE))
rmse_results

regu_user_movie_age_gen_mu

#VALIDATION


mu <- 3.712482

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+4.75))
  
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+4.75))
  
b_age <- edx %>%
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by ="userId") %>%
  group_by(Age) %>% 
  summarize(b_age = sum(rating - b_i - b_u - mu)/(n()+4.75))
  
b_gen <- edx %>% 
  left_join(b_i, by="movieId") %>% 
  left_join(b_u, by ="userId") %>%
  left_join(b_age, by = "Age") %>%
  group_by(genres) %>% 
  summarize(b_gen = sum(rating - b_i - b_u - b_age - mu)/(n()+4.75))
  
predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_age, by = "Age") %>%
  left_join(b_gen, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_age) %>%
  pull(pred)

RMSE(predicted_ratings, validation$rating)
