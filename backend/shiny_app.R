# MathBoard "Model & Activity" Shiny dashboard.
# Reads:
#   backend/activity.db                — live request log
#   backend/ml/artifacts/metrics.json  — training-time metrics
#   backend/ml/artifacts/confusion.json — training-time confusion matrix
#
# Run via: Rscript run_shiny.r

library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(DBI)
library(RSQLite)
library(jsonlite)

ACTIVITY_DB     <- "activity.db"
METRICS_JSON    <- "ml/artifacts/metrics.json"
CONFUSION_JSON  <- "ml/artifacts/confusion.json"
REFRESH_MS      <- 5000

read_activity <- function(limit = 50) {
  if (!file.exists(ACTIVITY_DB)) return(data.frame())
  con <- tryCatch(dbConnect(SQLite(), ACTIVITY_DB), error = function(e) NULL)
  if (is.null(con)) return(data.frame())
  on.exit(dbDisconnect(con))
  tryCatch(
    dbGetQuery(con, "SELECT * FROM requests ORDER BY id DESC LIMIT ?", params = list(limit)),
    error = function(e) data.frame()
  )
}

read_activity_all <- function() {
  if (!file.exists(ACTIVITY_DB)) return(data.frame())
  con <- tryCatch(dbConnect(SQLite(), ACTIVITY_DB), error = function(e) NULL)
  if (is.null(con)) return(data.frame())
  on.exit(dbDisconnect(con))
  tryCatch(dbGetQuery(con, "SELECT * FROM requests"), error = function(e) data.frame())
}

read_metrics <- function() {
  if (!file.exists(METRICS_JSON)) return(NULL)
  tryCatch(jsonlite::fromJSON(METRICS_JSON), error = function(e) NULL)
}

read_confusion <- function() {
  if (!file.exists(CONFUSION_JSON)) return(NULL)
  tryCatch(jsonlite::fromJSON(CONFUSION_JSON), error = function(e) NULL)
}

ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "MathBoard — Model & Activity"),
  dashboardSidebar(disable = TRUE),
  dashboardBody(
    tags$head(tags$style(HTML(
      ".content-wrapper { background: #f4f6f9; }
       .small-box .icon-large { right: 14px; }
       .badge-local { color: #4f46e5; font-weight: 700; }
       .badge-gemini { color: #64748b; font-weight: 700; }
       .agree-match { color: #15803d; font-weight: 700; }
       .agree-differ { color: #b45309; font-weight: 700; }
       .agree-na { color: #94a3b8; }
       .thumb { width: 48px; height: 48px; border: 1px solid #e6e8ef; border-radius: 4px; }"
    ))),
    fluidRow(
      valueBoxOutput("card_local_rate"),
      valueBoxOutput("card_accepted_acc"),
      valueBoxOutput("card_agreement"),
      valueBoxOutput("card_total")
    ),
    tabsetPanel(
      id = "main_tabs",
      tabPanel("Recent Activity", uiOutput("recent_table_ui")),
      tabPanel("Model Performance",
        fluidRow(
          box(title = "Confusion Matrix (top 30 by frequency)", width = 12, status = "primary",
              plotOutput("confusion_plot", height = "560px"))
        ),
        fluidRow(
          box(title = "Per-class accuracy (top 30)", width = 6, status = "info",
              plotOutput("perclass_plot", height = "420px")),
          box(title = "Confidence histogram (local model only)", width = 6, status = "info",
              plotOutput("conf_hist", height = "420px"))
        )
      ),
      tabPanel("Solver Agreement",
        fluidRow(
          box(title = "Daily agreement breakdown", width = 12, status = "warning",
              plotOutput("agree_over_time", height = "320px"))
        ),
        fluidRow(
          box(title = "Disagreements (SymPy != Ryacas)", width = 12, status = "danger",
              uiOutput("disagree_table_ui"))
        )
      ),
      tabPanel("Local vs Gemini",
        fluidRow(
          box(title = "Daily volume", width = 12, status = "primary",
              plotOutput("volume_plot", height = "320px"))
        ),
        fluidRow(
          box(title = "Top 10 handled locally", width = 6, status = "success",
              plotOutput("top_local", height = "320px")),
          box(title = "Top 10 falling through to Gemini", width = 6, status = "warning",
              plotOutput("top_gemini", height = "320px"))
        )
      )
    )
  )
)

server <- function(input, output, session) {
  tick <- reactiveTimer(REFRESH_MS)

  activity <- reactive({ tick(); read_activity_all() })
  recent <- reactive({ tick(); read_activity(limit = 50) })
  metrics <- reactive({ tick(); read_metrics() })
  confusion <- reactive({ tick(); read_confusion() })

  empty_card <- function(title) {
    valueBox("--", title, icon = icon("hourglass-half"), color = "light-blue")
  }

  output$card_local_rate <- renderValueBox({
    df <- activity()
    if (nrow(df) == 0) return(empty_card("Local hit rate"))
    rate <- mean(df$source == "local") * 100
    valueBox(sprintf("%.1f%%", rate), "Local hit rate",
             icon = icon("bolt"), color = "purple")
  })

  output$card_accepted_acc <- renderValueBox({
    m <- metrics()
    if (is.null(m)) return(empty_card("Accuracy on accepted"))
    valueBox(sprintf("%.1f%%", m$accuracy_on_accepted * 100),
             "Accuracy on accepted",
             icon = icon("bullseye"), color = "green")
  })

  output$card_agreement <- renderValueBox({
    df <- activity()
    if (nrow(df) == 0) return(empty_card("Solver agreement"))
    rate <- mean(df$agreement == "match", na.rm = TRUE) * 100
    valueBox(sprintf("%.1f%%", rate), "Solver agreement",
             icon = icon("check-double"), color = "yellow")
  })

  output$card_total <- renderValueBox({
    df <- activity()
    valueBox(format(nrow(df), big.mark = ","), "Total requests logged",
             icon = icon("list"), color = "blue")
  })

  output$recent_table_ui <- renderUI({
    df <- recent()
    if (nrow(df) == 0) {
      return(div(style = "padding: 20px; color: #64748b;",
                 "No requests logged yet. Use the Math Solver tab to generate data."))
    }
    rows <- lapply(seq_len(nrow(df)), function(i) {
      r <- df[i, ]
      thumb_src <- if (!is.na(r$thumbnail_b64)) {
        sprintf("data:image/png;base64,%s", r$thumbnail_b64)
      } else "data:,"
      src_class <- if (!is.na(r$source) && r$source == "local") "badge-local" else "badge-gemini"
      agree_class <- switch(as.character(r$agreement),
                            "match" = "agree-match",
                            "differ" = "agree-differ",
                            "agree-na")
      tags$tr(
        tags$td(format(as.POSIXct(r$timestamp, tz = "UTC"), "%H:%M:%S")),
        tags$td(tags$img(src = thumb_src, class = "thumb")),
        tags$td(tags$code(r$recognized_latex)),
        tags$td(class = src_class, toupper(r$source)),
        tags$td(if (!is.na(r$confidence)) sprintf("%.2f", r$confidence) else "--"),
        tags$td(if (!is.na(r$sympy_solution)) tags$code(r$sympy_solution) else "--"),
        tags$td(if (!is.na(r$ryacas_solution)) tags$code(r$ryacas_solution) else "--"),
        tags$td(class = agree_class, r$agreement)
      )
    })
    tags$table(
      class = "table table-striped table-condensed",
      tags$thead(tags$tr(
        tags$th("Time"), tags$th("Thumb"), tags$th("Recognized"),
        tags$th("Source"), tags$th("Conf"),
        tags$th("SymPy"), tags$th("Ryacas"), tags$th("Agreement")
      )),
      tags$tbody(rows)
    )
  })

  output$confusion_plot <- renderPlot({
    cf <- confusion()
    if (is.null(cf)) {
      plot.new()
      text(0.5, 0.5, "Local classifier not trained yet.\nRun python -m ml.train from backend/.", cex = 1.2)
      return()
    }
    classes <- unlist(cf$classes)
    mat <- as.matrix(cf$matrix)
    df <- expand.grid(true = classes, pred = classes, stringsAsFactors = FALSE)
    df$count <- as.vector(mat)
    df_top <- df %>% group_by(true) %>% summarise(total = sum(count)) %>%
      arrange(desc(total)) %>% head(30)
    df2 <- df %>% filter(true %in% df_top$true, pred %in% df_top$true)
    df2$true <- factor(df2$true, levels = df_top$true)
    df2$pred <- factor(df2$pred, levels = df_top$true)
    ggplot(df2, aes(pred, true, fill = count)) +
      geom_tile() +
      scale_fill_gradient(low = "white", high = "#4f46e5") +
      theme_minimal(base_size = 11) +
      theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
      labs(x = "Predicted", y = "True", fill = "Count")
  })

  output$perclass_plot <- renderPlot({
    cf <- confusion()
    if (is.null(cf)) { plot.new(); return() }
    classes <- unlist(cf$classes)
    mat <- as.matrix(cf$matrix)
    diag_vals <- diag(mat)
    totals <- rowSums(mat)
    acc <- ifelse(totals == 0, 0, diag_vals / totals)
    df <- data.frame(class = classes, accuracy = acc, count = totals) %>%
      arrange(desc(count)) %>% head(30)
    df$class <- factor(df$class, levels = rev(df$class))
    ggplot(df, aes(class, accuracy)) +
      geom_col(fill = "#4f46e5") +
      coord_flip() + theme_minimal(base_size = 11) +
      labs(x = NULL, y = "Per-class accuracy") +
      ylim(0, 1)
  })

  output$conf_hist <- renderPlot({
    df <- activity()
    if (nrow(df) == 0) { plot.new(); return() }
    df <- df %>% filter(source == "local", !is.na(confidence))
    if (nrow(df) == 0) { plot.new(); return() }
    ggplot(df, aes(confidence)) +
      geom_histogram(binwidth = 0.05, fill = "#4f46e5", boundary = 0) +
      scale_x_continuous(limits = c(0, 1)) + theme_minimal(base_size = 11) +
      labs(x = "Confidence", y = "Count")
  })

  output$agree_over_time <- renderPlot({
    df <- activity()
    if (nrow(df) == 0) { plot.new(); return() }
    df$date <- as.Date(df$timestamp)
    daily <- df %>% group_by(date, agreement) %>% summarise(n = n(), .groups = "drop")
    ggplot(daily, aes(date, n, fill = agreement)) +
      geom_bar(stat = "identity") +
      scale_fill_manual(values = c(
        match = "#15803d", differ = "#b45309",
        ryacas_unavailable = "#94a3b8", ryacas_error = "#dc2626"
      )) +
      theme_minimal(base_size = 12) + labs(x = NULL, y = "Requests", fill = "Agreement")
  })

  output$disagree_table_ui <- renderUI({
    df <- activity() %>% filter(agreement == "differ") %>% arrange(desc(timestamp)) %>% head(50)
    if (nrow(df) == 0) {
      return(div(style = "padding: 14px; color: #64748b;",
                 "No disagreements logged yet."))
    }
    tags$table(class = "table table-striped table-condensed",
      tags$thead(tags$tr(
        tags$th("Time"), tags$th("Recognized"),
        tags$th("SymPy"), tags$th("Ryacas")
      )),
      tags$tbody(lapply(seq_len(nrow(df)), function(i) {
        r <- df[i, ]
        tags$tr(
          tags$td(format(as.POSIXct(r$timestamp, tz = "UTC"), "%H:%M:%S")),
          tags$td(tags$code(r$recognized_latex)),
          tags$td(tags$code(r$sympy_solution)),
          tags$td(tags$code(r$ryacas_solution))
        )
      }))
    )
  })

  output$volume_plot <- renderPlot({
    df <- activity()
    if (nrow(df) == 0) { plot.new(); return() }
    df$date <- as.Date(df$timestamp)
    daily <- df %>% group_by(date, source) %>% summarise(n = n(), .groups = "drop")
    ggplot(daily, aes(date, n, color = source)) +
      geom_line(linewidth = 1) + geom_point(size = 2) +
      scale_color_manual(values = c(local = "#4f46e5", gemini = "#64748b")) +
      theme_minimal(base_size = 12) + labs(x = NULL, y = "Requests", color = "Source")
  })

  output$top_local <- renderPlot({
    df <- activity() %>% filter(source == "local")
    if (nrow(df) == 0) { plot.new(); return() }
    top <- df %>% count(recognized_latex, sort = TRUE) %>% head(10)
    top$recognized_latex <- factor(top$recognized_latex, levels = rev(top$recognized_latex))
    ggplot(top, aes(recognized_latex, n)) +
      geom_col(fill = "#15803d") + coord_flip() + theme_minimal(base_size = 11) +
      labs(x = NULL, y = "Count")
  })

  output$top_gemini <- renderPlot({
    df <- activity() %>% filter(source == "gemini")
    if (nrow(df) == 0) { plot.new(); return() }
    top <- df %>% count(recognized_latex, sort = TRUE) %>% head(10)
    top$recognized_latex <- factor(top$recognized_latex, levels = rev(top$recognized_latex))
    ggplot(top, aes(recognized_latex, n)) +
      geom_col(fill = "#b45309") + coord_flip() + theme_minimal(base_size = 11) +
      labs(x = NULL, y = "Count")
  })
}

shinyApp(ui, server, options = list(port = 3838, host = "0.0.0.0", launch.browser = FALSE))
