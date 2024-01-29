# Script to run Multiple Linear Regression
# Written by Jade Dubbeld
# 26/01/2024

# set working directory
setwd("~/Documents/Studie/Master Computational Science/Master Thesis/crime_communication_network/barabasi-albert")

# load data
data <- read.csv("data/measures-consensus-varying-graphs1-50.csv")

# model
model <- lm(consensus ~ link+betweenness+closeness+clustering+transitivity#,data = data)
            +# link*degree
            + link*betweenness
            + link*closeness
            + link*clustering
            + link*transitivity
            #+ link*global_efficiency
            #+ link*local_efficiency
            #+ degree*betweenness
            #+ degree*closeness
            #+ degree*clustering
            #+ degree*transitivity
            #+ degree*global_efficiency
            #+ degree*local_efficiency
            + betweenness*closeness 
            + betweenness*clustering
            + betweenness*transitivity
            #+ betweenness*global_efficiency
            #+ betweenness*local_efficiency
            + closeness*clustering
            + closeness*transitivity
            #+ closeness*global_efficiency
            #+ closeness*local_efficiency
            + clustering*transitivity
            #+ clustering*global_efficiency
            #+ clustering*local_efficiency
            #+ transitivity*global_efficiency
            #+ transitivity*local_efficiency
            #+ global_efficiency*local_efficiency
            ,data = data)
summary(model)

sigma(model)/mean(data$consensus)

coefficients(model)
anova(model)