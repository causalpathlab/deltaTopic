library(argparse)
parser <- ArgumentParser()
parser$add_argument("--SavePath",
    help = "relative path to save folder")
args <- parser$parse_args()

render_report = function(Save_Path) {
  rmarkdown::render(
    "report.Rmd", params = list(
      Save_Path = Save_Path
    ),
    output_file = paste0(Save_Path, "/Report.html")
  )
}

render_report(args$SavePath)
