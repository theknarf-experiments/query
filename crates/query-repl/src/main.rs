//! Query REPL - Interactive SQL and Datalog query interface

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use query::{Engine, QueryResult};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};
use std::io;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Sql,
    Datalog,
}

impl Mode {
    fn name(&self) -> &'static str {
        match self {
            Mode::Sql => "SQL",
            Mode::Datalog => "Datalog",
        }
    }

    fn toggle(&self) -> Self {
        match self {
            Mode::Sql => Mode::Datalog,
            Mode::Datalog => Mode::Sql,
        }
    }
}

struct App {
    engine: Engine,
    mode: Mode,
    input: String,
    cursor_pos: usize,
    history: Vec<HistoryEntry>,
    scroll_offset: usize,
    show_help: bool,
}

struct HistoryEntry {
    mode: Mode,
    query: String,
    result: String,
    is_error: bool,
}

impl App {
    fn new() -> Self {
        Self {
            engine: Engine::new(),
            mode: Mode::Sql,
            input: String::new(),
            cursor_pos: 0,
            history: Vec::new(),
            scroll_offset: 0,
            show_help: false,
        }
    }

    fn execute_query(&mut self) {
        if self.input.trim().is_empty() {
            return;
        }

        let query = self.input.clone();
        let (result, is_error) = match self.mode {
            Mode::Sql => match self.engine.execute(&query) {
                Ok(result) => (format_result(&result), false),
                Err(e) => (format!("Error: {:?}", e), true),
            },
            Mode::Datalog => match self.engine.execute_datalog(&query) {
                Ok(result) => (format_result(&result), false),
                Err(e) => (format!("Error: {:?}", e), true),
            },
        };

        self.history.push(HistoryEntry {
            mode: self.mode,
            query,
            result,
            is_error,
        });

        self.input.clear();
        self.cursor_pos = 0;
        self.scroll_offset = 0;
    }

    fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor_pos, c);
        self.cursor_pos += 1;
    }

    fn delete_char(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
            self.input.remove(self.cursor_pos);
        }
    }

    fn delete_char_forward(&mut self) {
        if self.cursor_pos < self.input.len() {
            self.input.remove(self.cursor_pos);
        }
    }

    fn move_cursor_left(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
        }
    }

    fn move_cursor_right(&mut self) {
        if self.cursor_pos < self.input.len() {
            self.cursor_pos += 1;
        }
    }

    fn move_cursor_start(&mut self) {
        self.cursor_pos = 0;
    }

    fn move_cursor_end(&mut self) {
        self.cursor_pos = self.input.len();
    }

    fn clear_input(&mut self) {
        self.input.clear();
        self.cursor_pos = 0;
    }
}

fn format_value(val: &query::logical::Value) -> String {
    use query::logical::Value;
    match val {
        Value::Null => "NULL".to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => format!("{}", f),
        Value::Text(s) => s.clone(),
        Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Value::Date(d) => d.to_string(),
        Value::Time(t) => t.to_string(),
        Value::Timestamp(ts) => ts.to_string(),
        Value::Json(j) => format!("{:?}", j),
    }
}

fn format_result(result: &QueryResult) -> String {
    match result {
        QueryResult::Select { columns, rows } => {
            if rows.is_empty() {
                return "(empty result set)".to_string();
            }

            // Calculate column widths
            let mut widths: Vec<usize> = columns.iter().map(|c| c.len()).collect();
            for row in rows {
                for (i, val) in row.iter().enumerate() {
                    if i < widths.len() {
                        widths[i] = widths[i].max(format_value(val).len());
                    }
                }
            }

            let mut output = String::new();

            // Header
            let header: Vec<String> = columns
                .iter()
                .zip(&widths)
                .map(|(c, w)| format!("{:width$}", c, width = *w))
                .collect();
            output.push_str(&header.join(" | "));
            output.push('\n');

            // Separator
            let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
            output.push_str(&sep.join("-+-"));
            output.push('\n');

            // Rows
            for row in rows {
                let formatted: Vec<String> = row
                    .iter()
                    .zip(&widths)
                    .map(|(v, w)| format!("{:width$}", format_value(v), width = *w))
                    .collect();
                output.push_str(&formatted.join(" | "));
                output.push('\n');
            }

            output.push_str(&format!(
                "\n({} row{})",
                rows.len(),
                if rows.len() == 1 { "" } else { "s" }
            ));
            output
        }
        QueryResult::RowsAffected(count) => format!("{} row(s) affected", count),
        QueryResult::Success => "OK".to_string(),
        QueryResult::TransactionStarted => "BEGIN".to_string(),
        QueryResult::TransactionCommitted => "COMMIT".to_string(),
        QueryResult::TransactionRolledBack => "ROLLBACK".to_string(),
        QueryResult::SavepointCreated(name) => format!("SAVEPOINT {}", name),
        QueryResult::SavepointReleased(name) => format!("RELEASE SAVEPOINT {}", name),
        QueryResult::RolledBackToSavepoint(name) => format!("ROLLBACK TO {}", name),
    }
}

fn main() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let result = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        if let Event::Key(key) = event::read()? {
            if app.show_help {
                app.show_help = false;
                continue;
            }

            match (key.code, key.modifiers) {
                // Exit
                (KeyCode::Char('c'), KeyModifiers::CONTROL) => return Ok(()),
                (KeyCode::Char('d'), KeyModifiers::CONTROL) if app.input.is_empty() => {
                    return Ok(());
                }

                // Toggle mode
                (KeyCode::Tab, _) => app.mode = app.mode.toggle(),

                // Execute query
                (KeyCode::Enter, _) => app.execute_query(),

                // Help
                (KeyCode::Char('?'), KeyModifiers::CONTROL) | (KeyCode::F(1), _) => {
                    app.show_help = true;
                }

                // Cursor movement
                (KeyCode::Left, _) => app.move_cursor_left(),
                (KeyCode::Right, _) => app.move_cursor_right(),
                (KeyCode::Home, _) | (KeyCode::Char('a'), KeyModifiers::CONTROL) => {
                    app.move_cursor_start()
                }
                (KeyCode::End, _) | (KeyCode::Char('e'), KeyModifiers::CONTROL) => {
                    app.move_cursor_end()
                }

                // Editing
                (KeyCode::Backspace, _) => app.delete_char(),
                (KeyCode::Delete, _) => app.delete_char_forward(),
                (KeyCode::Char('u'), KeyModifiers::CONTROL) => app.clear_input(),

                // Scrolling history
                (KeyCode::Up, _) | (KeyCode::PageUp, _) => {
                    if app.scroll_offset < app.history.len().saturating_sub(1) {
                        app.scroll_offset += 1;
                    }
                }
                (KeyCode::Down, _) | (KeyCode::PageDown, _) => {
                    app.scroll_offset = app.scroll_offset.saturating_sub(1);
                }

                // Character input
                (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                    app.insert_char(c);
                }

                _ => {}
            }
        }
    }
}

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Title bar
            Constraint::Min(5),    // History/results
            Constraint::Length(3), // Input
            Constraint::Length(1), // Status bar
        ])
        .split(f.area());

    // Title bar
    let title = Line::from(vec![
        Span::raw(" Query REPL "),
        Span::styled(
            format!("[{}]", app.mode.name()),
            Style::default()
                .fg(match app.mode {
                    Mode::Sql => Color::Cyan,
                    Mode::Datalog => Color::Magenta,
                })
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" - Press Tab to switch modes, F1 for help"),
    ]);
    let title_bar =
        Paragraph::new(title).style(Style::default().bg(Color::DarkGray).fg(Color::White));
    f.render_widget(title_bar, chunks[0]);

    // History area
    render_history(f, app, chunks[1]);

    // Input area
    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(match app.mode {
            Mode::Sql => Color::Cyan,
            Mode::Datalog => Color::Magenta,
        }))
        .title(format!(" {} Query ", app.mode.name()));

    let input_area = input_block.inner(chunks[2]);
    f.render_widget(input_block, chunks[2]);

    let input_text = Paragraph::new(app.input.as_str());
    f.render_widget(input_text, input_area);

    // Position cursor
    f.set_cursor_position((input_area.x + app.cursor_pos as u16, input_area.y));

    // Status bar
    let status = Line::from(vec![
        Span::styled(" Ctrl+C ", Style::default().bg(Color::DarkGray)),
        Span::raw(" Exit "),
        Span::styled(" Tab ", Style::default().bg(Color::DarkGray)),
        Span::raw(" Switch mode "),
        Span::styled(" Enter ", Style::default().bg(Color::DarkGray)),
        Span::raw(" Execute "),
        Span::styled(" ↑/↓ ", Style::default().bg(Color::DarkGray)),
        Span::raw(" Scroll "),
    ]);
    let status_bar = Paragraph::new(status);
    f.render_widget(status_bar, chunks[3]);

    // Help popup
    if app.show_help {
        render_help_popup(f);
    }
}

fn render_history(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" History ");

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.history.is_empty() {
        let welcome = Text::from(vec![
            Line::from(""),
            Line::from("  Welcome to Query REPL!"),
            Line::from(""),
            Line::from("  Enter SQL or Datalog queries below."),
            Line::from("  Press Tab to switch between modes."),
            Line::from(""),
            Line::from("  Example SQL:"),
            Line::styled(
                "    CREATE TABLE users (id INT, name TEXT)",
                Style::default().fg(Color::Cyan),
            ),
            Line::styled(
                "    INSERT INTO users VALUES (1, 'Alice')",
                Style::default().fg(Color::Cyan),
            ),
            Line::styled("    SELECT * FROM users", Style::default().fg(Color::Cyan)),
            Line::from(""),
            Line::from("  Example Datalog:"),
            Line::styled(
                "    path(X, Y) :- edge(X, Y).",
                Style::default().fg(Color::Magenta),
            ),
            Line::styled(
                "    path(X, Z) :- path(X, Y), edge(Y, Z).",
                Style::default().fg(Color::Magenta),
            ),
            Line::styled("    ?- path(1, X).", Style::default().fg(Color::Magenta)),
        ]);
        let para = Paragraph::new(welcome);
        f.render_widget(para, inner);
        return;
    }

    let mut lines: Vec<Line> = Vec::new();

    for entry in app.history.iter().rev().skip(app.scroll_offset) {
        // Query line
        let mode_color = match entry.mode {
            Mode::Sql => Color::Cyan,
            Mode::Datalog => Color::Magenta,
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("[{}] ", entry.mode.name()),
                Style::default().fg(mode_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(&entry.query, Style::default().fg(Color::Yellow)),
        ]));

        // Result lines
        let result_style = if entry.is_error {
            Style::default().fg(Color::Red)
        } else {
            Style::default().fg(Color::Green)
        };

        for line in entry.result.lines() {
            lines.push(Line::styled(format!("  {}", line), result_style));
        }

        lines.push(Line::from(""));
    }

    let para = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(para, inner);
}

fn render_help_popup(f: &mut Frame) {
    let area = centered_rect(60, 70, f.area());

    f.render_widget(Clear, area);

    let help_text = vec![
        Line::from("").centered(),
        Line::styled(
            "Query REPL Help",
            Style::default().add_modifier(Modifier::BOLD),
        )
        .centered(),
        Line::from("").centered(),
        Line::from(""),
        Line::from("  Keybindings:"),
        Line::from(""),
        Line::from("    Tab          Switch between SQL and Datalog"),
        Line::from("    Enter        Execute query"),
        Line::from("    Ctrl+C       Exit"),
        Line::from("    Ctrl+D       Exit (when input is empty)"),
        Line::from("    ↑/↓          Scroll through history"),
        Line::from("    Ctrl+A/Home  Move cursor to start"),
        Line::from("    Ctrl+E/End   Move cursor to end"),
        Line::from("    Ctrl+U       Clear input"),
        Line::from("    F1           Show this help"),
        Line::from(""),
        Line::from("  SQL Mode:"),
        Line::from("    Standard SQL queries (SELECT, INSERT, etc.)"),
        Line::from(""),
        Line::from("  Datalog Mode:"),
        Line::from("    Logic programming with rules and queries"),
        Line::from("    Facts: predicate(arg1, arg2)."),
        Line::from("    Rules: head(X) :- body(X), other(X)."),
        Line::from("    Query: ?- predicate(X)."),
        Line::from(""),
        Line::styled(
            "  Press any key to close",
            Style::default().fg(Color::DarkGray),
        )
        .centered(),
    ];

    let help = Paragraph::new(help_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" Help "),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(help, area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
