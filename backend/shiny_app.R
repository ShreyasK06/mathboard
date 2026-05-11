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
  skin = "black",
  dashboardHeader(title = "MathBoard — Activity"),
  dashboardSidebar(disable = TRUE),
  dashboardBody(
    tags$head(
      tags$link(rel = "preconnect", href = "https://fonts.googleapis.com"),
      tags$link(rel = "stylesheet",
        href = "https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"),
      tags$style(HTML("
        :root {
          --paper:   #f1ece1;
          --paper-2: #e8e2d3;
          --paper-3: #ddd5c2;
          --ink:     #14130f;
          --ink-2:   #2a2723;
          --ink-3:   #6b6557;
          --ink-4:   #97907f;
          --rule:    #c8c0ad;
          --accent:  #d8ff3a;
          --serif:   'Instrument Serif', Georgia, serif;
          --sans:    'Geist', system-ui, sans-serif;
          --mono:    'JetBrains Mono', monospace;
        }

        body, .content-wrapper, .main-sidebar, .main-header, input, select, textarea, button {
          font-family: var(--sans) !important;
          font-size: 14px !important;
        }

        body { background: var(--paper) !important; }
        .content-wrapper { background: var(--paper) !important; }
        .content { padding: 20px !important; }

        .main-header .logo {
          background: var(--ink) !important;
          color: var(--paper) !important;
          font-family: var(--serif) !important;
          font-style: italic !important;
          font-size: 1rem !important;
          letter-spacing: 0 !important;
          border-bottom: none !important;
        }
        .main-header .navbar { background: var(--ink) !important; border-bottom: none !important; }
        .main-header .logo:hover { background: var(--ink-2) !important; }
        .main-header .navbar .sidebar-toggle { color: rgba(241,236,225,.6) !important; }
        .main-header .navbar .sidebar-toggle:hover { color: var(--paper) !important; background: transparent !important; }

        .small-box {
          border-radius: 4px !important;
          box-shadow: none !important;
          border: 1px solid var(--rule) !important;
          background: var(--paper-2) !important;
          color: var(--ink) !important;
        }
        .small-box h3 {
          font-family: var(--serif) !important;
          font-style: italic !important;
          font-size: 2.2rem !important;
          font-weight: 400 !important;
          letter-spacing: -0.02em !important;
          color: var(--ink) !important;
        }
        .small-box p {
          font-family: var(--mono) !important;
          font-size: 0.68rem !important;
          font-weight: 400 !important;
          text-transform: uppercase !important;
          letter-spacing: 0.08em !important;
          color: var(--ink-3) !important;
          opacity: 1 !important;
        }
        .small-box .icon-large { right: 14px !important; color: var(--rule) !important; opacity: 1 !important; }

        .box {
          border-radius: 4px !important;
          border: 1px solid var(--rule) !important;
          box-shadow: none !important;
          background: var(--paper-2) !important;
        }
        .box-header {
          border-bottom: 1px solid var(--rule) !important;
          padding: 12px 16px !important;
          background: var(--paper-2) !important;
        }
        .box-title {
          font-family: var(--mono) !important;
          font-size: 0.7rem !important;
          font-weight: 400 !important;
          letter-spacing: 0.08em !important;
          text-transform: uppercase !important;
          color: var(--ink-3) !important;
        }
        .box-body { padding: 16px !important; background: var(--paper) !important; }
        .box.box-primary > .box-header,
        .box.box-info    > .box-header,
        .box.box-success > .box-header,
        .box.box-warning > .box-header,
        .box.box-danger  > .box-header { border-top: none !important; }

        .nav-tabs-custom {
          border-radius: 4px !important;
          box-shadow: none !important;
          border: 1px solid var(--rule) !important;
          background: var(--paper) !important;
        }
        .nav-tabs-custom > .nav-tabs { background: var(--paper-2) !important; border-bottom: 1px solid var(--rule) !important; }
        .nav-tabs-custom > .nav-tabs > li > a {
          font-family: var(--mono) !important;
          font-size: 0.72rem !important;
          font-weight: 400 !important;
          letter-spacing: 0.06em !important;
          text-transform: uppercase !important;
          color: var(--ink-4) !important;
          border: none !important;
          border-radius: 0 !important;
        }
        .nav-tabs-custom > .nav-tabs > li.active > a {
          color: var(--ink) !important;
          border-top: 2px solid var(--ink) !important;
          background: var(--paper) !important;
          font-weight: 600 !important;
        }
        .nav-tabs-custom > .tab-content { padding: 16px !important; background: var(--paper) !important; }

        .table { font-size: 0.82rem !important; color: var(--ink) !important; background: var(--paper) !important; }
        .table > thead > tr > th {
          font-family: var(--mono) !important;
          font-size: 0.68rem !important;
          font-weight: 400 !important;
          text-transform: uppercase !important;
          letter-spacing: 0.08em !important;
          color: var(--ink-4) !important;
          border-bottom: 1px solid var(--rule) !important;
          padding: 8px 12px !important;
          background: var(--paper-2) !important;
        }
        .table > tbody > tr > td { padding: 10px 12px !important; vertical-align: middle !important; border-color: var(--rule) !important; color: var(--ink) !important; }
        .table-striped > tbody > tr:nth-child(odd) > td { background: var(--paper-2) !important; }
        .table-condensed > thead > tr > th, .table-condensed > tbody > tr > td { padding: 8px 12px !important; }

        code {
          font-family: var(--mono) !important;
          font-size: 0.78rem !important;
          background: rgba(20,19,15,0.07) !important;
          color: var(--ink-2) !important;
          padding: 2px 6px !important;
          border-radius: 3px !important;
          border: none !important;
        }

        .badge-local {
          display: inline-block; color: var(--ink); font-family: var(--mono);
          font-weight: 600; background: var(--accent); padding: 2px 9px;
          border-radius: 20px; font-size: 0.7rem; letter-spacing: 0.04em;
        }
        .badge-gemini {
          display: inline-block; color: var(--ink-3); font-family: var(--mono);
          font-weight: 400; background: var(--paper-3); padding: 2px 9px;
          border-radius: 20px; font-size: 0.7rem; letter-spacing: 0.04em;
          border: 1px solid var(--rule);
        }
        .agree-match  { color: #15803d; font-weight: 600; }
        .agree-differ { color: #b45309; font-weight: 600; }
        .agree-na     { color: var(--ink-4); }

        .thumb { width: 44px; height: 44px; border: 1px solid var(--rule); border-radius: 4px; object-fit: contain; background: var(--paper); }

        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--rule); border-radius: 3px; }
      "))
    ),
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
          box(title = "Disagreements (Primary != Gemini)", width = 12, status = "danger",
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
      primary_label <- if (is.null(r$primary_solver) || is.na(r$primary_solver)) "--" else toupper(r$primary_solver)
      tags$tr(
        tags$td(format(as.POSIXct(r$timestamp, tz = "UTC"), "%H:%M:%S")),
        tags$td(tags$img(src = thumb_src, class = "thumb")),
        tags$td(tags$code(r$recognized_latex)),
        tags$td(class = src_class, toupper(r$source)),
        tags$td(if (!is.na(r$confidence)) sprintf("%.2f", r$confidence) else "--"),
        tags$td(primary_label),
        tags$td(if (!is.na(r$primary_solution)) tags$code(r$primary_solution) else "--"),
        tags$td(if (!is.na(r$crosscheck_solution)) tags$code(r$crosscheck_solution) else "--"),
        tags$td(class = agree_class, r$agreement)
      )
    })
    tags$table(
      class = "table table-striped table-condensed",
      tags$thead(tags$tr(
        tags$th("Time"), tags$th("Thumb"), tags$th("Recognized"),
        tags$th("Source"), tags$th("Conf"),
        tags$th("Primary engine"), tags$th("Primary"),
        tags$th("Gemini cross-check"), tags$th("Agreement")
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
      scale_fill_gradient(low = "#f1ece1", high = "#14130f") +
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
      geom_col(fill = "#14130f") +
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
      geom_histogram(binwidth = 0.05, fill = "#14130f", boundary = 0) +
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
        crosscheck_unavailable = "#94a3b8", crosscheck_error = "#dc2626",
        # Tolerate rows from before the rename so old data still plots.
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
        tags$th("Primary"), tags$th("Gemini cross-check")
      )),
      tags$tbody(lapply(seq_len(nrow(df)), function(i) {
        r <- df[i, ]
        tags$tr(
          tags$td(format(as.POSIXct(r$timestamp, tz = "UTC"), "%H:%M:%S")),
          tags$td(tags$code(r$recognized_latex)),
          tags$td(tags$code(r$primary_solution)),
          tags$td(tags$code(r$crosscheck_solution))
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
      scale_color_manual(values = c(local = "#14130f", gemini = "#97907f")) +
      theme_minimal(base_size = 12) + labs(x = NULL, y = "Requests", color = "Source")
  })

  output$top_local <- renderPlot({
    df <- activity() %>% filter(source == "local")
    if (nrow(df) == 0) { plot.new(); return() }
    top <- df %>% count(recognized_latex, sort = TRUE) %>% head(10)
    top$recognized_latex <- factor(top$recognized_latex, levels = rev(top$recognized_latex))
    ggplot(top, aes(recognized_latex, n)) +
      geom_col(fill = "#14130f") + coord_flip() + theme_minimal(base_size = 11) +
      labs(x = NULL, y = "Count")
  })

  output$top_gemini <- renderPlot({
    df <- activity() %>% filter(source == "gemini")
    if (nrow(df) == 0) { plot.new(); return() }
    top <- df %>% count(recognized_latex, sort = TRUE) %>% head(10)
    top$recognized_latex <- factor(top$recognized_latex, levels = rev(top$recognized_latex))
    ggplot(top, aes(recognized_latex, n)) +
      geom_col(fill = "#97907f") + coord_flip() + theme_minimal(base_size = 11) +
      labs(x = NULL, y = "Count")
  })
}

shinyApp(ui, server, options = list(port = 3838, host = "0.0.0.0", launch.browser = FALSE))
