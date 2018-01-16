#!/usr/bin/env Rscript
#require(gridExtra)
#require(reshape)
#require(ggplot2)

source("vioplot.R")
require(plotrix)

args = commandArgs(trailingOnly=TRUE)

cache_min_color="deepskyblue"
cache_max_color="gold"
hbm_min_color="deepskyblue"
hbm_max_color="gold"


# test if there is at least one argument: if not, return an error
if (length(args)!=2) {
  stop("Please provide CACHE_MODE_FILE and HBM_FILE, e.g., R < foo.R cache.csv hbm.csv.", call.=FALSE)
} else if (length(args)==2) {
  # default output file
  cache_mode_file=args[1]
  hbm_mode_file=args[2]
}

cat('Cache mode: ', cache_mode_file, '\n')
cat('HBM mode: ', hbm_mode_file, '\n')

fname_base=sub(pattern = "cache_", replacement = "", tools::file_path_sans_ext(basename(cache_mode_file)))

fname=sprintf('%s.pdf', fname_base)
print(fname)
pdf(file=fname, width = 11, height = 8.5)

all_cache_df <- read.csv(cache_mode_file, header=TRUE)
all_flat_hbm_df <- read.csv(hbm_mode_file, header=TRUE)

all_cache_df['mode'] = 'Cache'
all_flat_hbm_df['mode'] = 'HBM'

#timer_names <- as.character(unique(unlist(all_flat_hbm_df['timer_name'])))
#timer_names <- unique(all_flat_hbm_df['timer_name'])
#print(timer_names)
#print(length(timer_names))


# tapply
timer_names <- sort(tapply(all_cache_df$maxT, all_cache_df$timer_name, max), decreasing=TRUE)
num_timers=length(timer_names)

# get the descriptive stats
#d_stats['minT'] = tapply(df$minT, df$timer_name, summary)
#d_stats['maxT'] = tapply(df$maxT, df$timer_name, summary)

join_cols <- c('mode', 'minT', 'maxT')

x1 = c(0)
xend = c(0)
y1 = c(0)
yend = c(0)

#p <- c(0,0,0,0)
p <- data.frame( x1,xend,y1,yend )
print(p)

for (timer_name in names(timer_names))
{
  #timer_name=timer_names[i]
  print(timer_name)
  flat_hbm_df <- all_flat_hbm_df[all_flat_hbm_df['timer_name'] == timer_name, ]
  cache_df <- all_cache_df[all_cache_df['timer_name'] == timer_name, ]


  summary(flat_hbm_df$minT)
  summary(flat_hbm_df$maxT)

  length(flat_hbm_df$minT)

  summary(cache_df$minT)
  summary(cache_df$maxT)

  length(cache_df$maxT)

  timer_name=cache_df$timer[1]
  timer_name=gsub("::", ":", timer_name)
  timer_name=gsub(":", "_", timer_name)

  all_data <- rbind( flat_hbm_df[join_cols], cache_df[join_cols] )
  d_stats_min <- tapply(all_data$minT, all_data$mode, summary)
  d_stats_max <- tapply(all_data$maxT, all_data$mode, summary)

  print(d_stats_min$Cache)

  # cache mode Q1
  p$y1 = d_stats_min$Cache['1st Qu.']
  p$yend = d_stats_min$Cache['1st Qu.']
  p$x1 = 1
  p$xend = 3
  print(p)
  #p3 <- p2 %+% geom_segment(aes(size=10), data = p)

  v <- vioplot(cache_df$minT,
          flat_hbm_df$minT,
          c(d_stats_min$Cache['1st Qu.']),
          names=c("Cache (minT)", "HBM (minT)", ""),
          col=c(cache_min_color, hbm_min_color, "white"))
  # 
  segments(1, d_stats_min$Cache['1st Qu.'],
           3-0.25, d_stats_min$Cache['1st Qu.'], col='blue')
  segments(2, d_stats_min$HBM['1st Qu.'],
           3-0.25, d_stats_min$HBM['1st Qu.'], col='blue')

  midpoint <- (d_stats_min$Cache['1st Qu.']-d_stats_min$HBM['1st Qu.'])/2.0 + d_stats_min$HBM['1st Qu.']

  segments(3-0.25, d_stats_min$Cache['1st Qu.'],
           3-0.25, d_stats_min$HBM['1st Qu.'], col='blue')

  boxed.labels(3-0.25,midpoint, c(sprintf('%2.2f%%',abs(d_stats_min$Cache['1st Qu.']-d_stats_min$HBM['1st Qu.'])*100.0/d_stats_min$HBM['1st Qu.'])), cex=1.5)
 

  # offset
  o = 0.40 
  segments(1, d_stats_min$Cache['3rd Qu.'],
           3+o, d_stats_min$Cache['3rd Qu.'], col='green')
  segments(2, d_stats_min$HBM['3rd Qu.'],
           3+o, d_stats_min$HBM['3rd Qu.'], col='green')

  midpoint <- (d_stats_min$Cache['3rd Qu.']-d_stats_min$HBM['3rd Qu.'])/2.0 + d_stats_min$HBM['3rd Qu.']

  segments(3+o, d_stats_min$Cache['3rd Qu.'],
           3+o, d_stats_min$HBM['3rd Qu.'], col='green')

  boxed.labels(3+o,midpoint, c(sprintf('%2.2f%%',abs(d_stats_min$Cache['3rd Qu.']-d_stats_min$HBM['3rd Qu.'])*100.0/d_stats_min$HBM['3rd Qu.'])), cex=1.5)

  # offset
  o = 0.0
  segments(1, d_stats_min$Cache['Median'],
           3+o, d_stats_min$Cache['Median'], col='orange')
  segments(2, d_stats_min$HBM['Median'],
           3+o, d_stats_min$HBM['Median'], col='orange')

  midpoint <- (d_stats_min$Cache['Median']-d_stats_min$HBM['Median'])/2.0 + d_stats_min$HBM['Median']

  segments(3+o, d_stats_min$Cache['Median'],
           3+o, d_stats_min$HBM['Median'], col='orange')

  boxed.labels(3+o,midpoint, c(sprintf('%2.2f%%',abs(d_stats_min$Cache['Median']-d_stats_min$HBM['Median'])*100.0/d_stats_min$HBM['Median'])), cex=1.5)



#  vioplot(cache_df$minT,
#          flat_hbm_df$minT,
#          cache_df$maxT,
#          flat_hbm_df$maxT,
#          names=c("Cache (minT)", "HBM (minT)", "Cache (maxT)", "HBM (maxT)"),
#          col=c(cache_min_color, hbm_min_color, cache_max_color, hbm_max_color))
  title(sprintf("Cache and Flat (pinned) HBM comparison of\n%s\nusing 64 procs per node with 4 HTs per proc (OpenMP)", cache_df$timer[1]))
  #break
}

dev.off()


