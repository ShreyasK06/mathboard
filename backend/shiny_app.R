library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)

DATA_DIR  <- "data/hasy"
IMAGE_DIR <- file.path(DATA_DIR, "hasy-data")
LABELS_CSV <- file.path(DATA_DIR, "hasy-data-labels.csv")

# Load dataset at startup — shows error message if dataset not downloaded yet
if (file.exists(LABELS_CSV)) {
  labels <- read.csv(LABELS_CSV, stringsAsFactors = FALSE)
  freq <- labels %>%
    group_by(latex) %>%
    summarise(count = n(), .groups = "drop") %>%
    arrange(desc(count))
  dataset_loaded <- TRUE
} else {
  labels <- data.frame()
  freq   <- data.frame(latex = character(), count = integer())
  dataset_loaded <- FALSE
}

if (dir.exists(IMAGE_DIR)) {
  addResourcePath("hasy-images", IMAGE_DIR)
}

ui <- dashboardPage(
  skin = "purple",
  dashboardHeader(title = "HASYv2 Math Symbol Explorer"),
  dashboardSidebar(disable = TRUE),
  dashboardBody(
    tags$head(tags$style(HTML("
      .content-wrapper { background: #f4f6f9; }
      .small-box { border-radius: 10px; }
      .sample-img {
        width: 64px; height: 64px; margin: 4px;
        image-rendering: pixelated;
        border: 1px solid #ddd; border-radius: 4px;
      }
    "))),

    if (!dataset_loaded) {
      fluidRow(
        box(
          width = 12, status = "warning", title = "Dataset not found",
          p("HASYv2 dataset not found at", code("backend/data/hasy/hasy-data-labels.csv")),
          p("Search 'HASYv2 dataset zenodo' and download HASYv2.tar.bz2, then extract to:"),
          tags$ul(
            tags$li(code("backend/data/hasy/hasy-data/"),"  — all PNG files"),
            tags$li(code("backend/data/hasy/hasy-data-labels.csv"))
          ),
          p("Restart the Shiny app after placing the files.")
        )
      )
    } else {
      tagList(
        fluidRow(
          valueBox(
            format(nrow(labels), big.mark = ","), "Total Samples",
            icon = icon("images"), color = "purple"
          ),
          valueBox(
            nrow(freq), "Symbol Classes",
            icon = icon("shapes"), color = "green"
          ),
          valueBox(
            "32 × 32", "Image Size (px)",
            icon = icon("expand-arrows-alt"), color = "yellow"
          )
        ),
        fluidRow(
          box(
            title = "Symbol Frequency Distribution", status = "primary",
            solidHeader = TRUE, width = 6,
            sliderInput("top_n", "Show top N classes:",
                        min = 10, max = 50, value = 20, step = 5),
            plotOutput("freq_chart", height = "380px")
          ),
          box(
            title = "Symbol Samples", status = "success",
            solidHeader = TRUE, width = 6,
            selectInput("symbol_class", "Select symbol class:",
                        choices = freq$latex, selected = freq$latex[1]),
            uiOutput("sample_grid")
          )
        )
      )
    }
  )
)

server <- function(input, output, session) {

  output$freq_chart <- renderPlot({
    req(dataset_loaded)
    top <- freq %>% head(input$top_n)
    ggplot(top, aes(x = reorder(latex, count), y = count)) +
      geom_bar(stat = "identity", fill = "#667eea", width = 0.7) +
      coord_flip() +
      labs(
        x = "Symbol (LaTeX label)",
        y = "Number of samples",
        title = paste("Top", input$top_n, "Most Frequent Symbols")
      ) +
      theme_minimal(base_size = 13) +
      theme(
        plot.title = element_text(face = "bold", color = "#444"),
        axis.text.y = element_text(size = 10)
      )
  })

  output$sample_grid <- renderUI({
    req(dataset_loaded, input$symbol_class)
    selected_rows <- labels %>% filter(latex == input$symbol_class)
    n_samples <- min(9, nrow(selected_rows))
    samples <- selected_rows %>% sample_n(n_samples)

    img_tags <- lapply(seq_len(nrow(samples)), function(i) {
      filename <- basename(samples$path[i])
      tags$img(
        src   = paste0("/hasy-images/", filename),
        class = "sample-img",
        title = samples$latex[i]
      )
    })

    tagList(
      p(
        style = "color:#888; font-size:13px; margin-bottom:8px;",
        paste(nrow(selected_rows), "total samples for symbol:", input$symbol_class)
      ),
      div(style = "display:flex; flex-wrap:wrap;", img_tags)
    )
  })
}

shinyApp(ui, server, options = list(port = 3838, host = "0.0.0.0", launch.browser = FALSE))
