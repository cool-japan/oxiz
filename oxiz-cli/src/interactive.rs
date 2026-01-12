//! Interactive mode (REPL) for OxiZ SMT Solver
//!
//! This module provides an interactive Read-Eval-Print Loop (REPL) for the OxiZ SMT solver,
//! featuring syntax highlighting, command history, auto-completion, and multi-line input support.

use std::path::PathBuf;
use std::time::Instant;

use owo_colors::{OwoColorize, Stream};
use oxiz_solver::Context;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::FileHistory;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{CompletionType, Config, Editor, Helper};

use crate::execute_and_format;
use crate::format::{eprintln_colored, print_help, print_version_info, println_colored};
use crate::{Args, Verbosity};

/// SMT-LIB2 syntax highlighter for the interactive REPL
struct SmtHighlighter {
    enable_colors: bool,
}

impl SmtHighlighter {
    fn new(enable_colors: bool) -> Self {
        Self { enable_colors }
    }

    fn is_keyword(word: &str) -> bool {
        matches!(
            word,
            "set-logic"
                | "declare-const"
                | "declare-fun"
                | "declare-sort"
                | "define-fun"
                | "define-sort"
                | "assert"
                | "check-sat"
                | "get-model"
                | "get-value"
                | "get-proof"
                | "get-unsat-core"
                | "get-assignment"
                | "get-assertions"
                | "get-info"
                | "get-option"
                | "set-info"
                | "set-option"
                | "push"
                | "pop"
                | "exit"
                | "echo"
                | "reset"
                | "reset-assertions"
                | "let"
                | "forall"
                | "exists"
                | "match"
                | "as"
                | "!"
                | "_"
        )
    }

    fn is_builtin_op(word: &str) -> bool {
        matches!(
            word,
            "+" | "-"
                | "*"
                | "/"
                | "div"
                | "mod"
                | "abs"
                | "="
                | "<"
                | ">"
                | "<="
                | ">="
                | "and"
                | "or"
                | "not"
                | "xor"
                | "=>"
                | "iff"
                | "ite"
                | "distinct"
        )
    }

    fn is_sort(word: &str) -> bool {
        matches!(
            word,
            "Bool" | "Int" | "Real" | "String" | "Array" | "BitVec"
        )
    }

    fn is_constant(word: &str) -> bool {
        word == "true" || word == "false"
    }
}

impl Highlighter for SmtHighlighter {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> std::borrow::Cow<'l, str> {
        if !self.enable_colors {
            return std::borrow::Cow::Borrowed(line);
        }

        let mut result = String::new();
        let mut word = String::new();
        let mut in_string = false;
        let mut in_comment = false;

        for ch in line.chars() {
            if in_comment {
                result.push_str(&format!(
                    "{}",
                    ch.to_string()
                        .if_supports_color(Stream::Stdout, |t| t.bright_black())
                ));
                continue;
            }

            if in_string {
                result.push_str(&format!(
                    "{}",
                    ch.to_string()
                        .if_supports_color(Stream::Stdout, |t| t.green())
                ));
                if ch == '"' {
                    in_string = false;
                }
                continue;
            }

            match ch {
                ';' => {
                    if !word.is_empty() {
                        result.push_str(&word);
                        word.clear();
                    }
                    in_comment = true;
                    result.push_str(&format!(
                        "{}",
                        ch.to_string()
                            .if_supports_color(Stream::Stdout, |t| t.bright_black())
                    ));
                }
                '"' => {
                    if !word.is_empty() {
                        result.push_str(&word);
                        word.clear();
                    }
                    in_string = true;
                    result.push_str(&format!(
                        "{}",
                        ch.to_string()
                            .if_supports_color(Stream::Stdout, |t| t.green())
                    ));
                }
                '(' | ')' => {
                    if !word.is_empty() {
                        let colored = if Self::is_keyword(&word) {
                            word.if_supports_color(Stream::Stdout, |t| t.blue())
                                .to_string()
                        } else if Self::is_builtin_op(&word) {
                            word.if_supports_color(Stream::Stdout, |t| t.magenta())
                                .to_string()
                        } else if Self::is_sort(&word) {
                            word.if_supports_color(Stream::Stdout, |t| t.cyan())
                                .to_string()
                        } else if Self::is_constant(&word) {
                            word.if_supports_color(Stream::Stdout, |t| t.yellow())
                                .to_string()
                        } else if word.chars().all(|c| c.is_ascii_digit() || c == '-') {
                            word.if_supports_color(Stream::Stdout, |t| t.yellow())
                                .to_string()
                        } else {
                            word.clone()
                        };
                        result.push_str(&colored);
                        word.clear();
                    }
                    result.push_str(&format!(
                        "{}",
                        ch.to_string()
                            .if_supports_color(Stream::Stdout, |t| t.bright_white())
                    ));
                }
                ' ' | '\t' | '\n' | '\r' => {
                    if !word.is_empty() {
                        let colored = if Self::is_keyword(&word) {
                            word.if_supports_color(Stream::Stdout, |t| t.blue())
                                .to_string()
                        } else if Self::is_builtin_op(&word) {
                            word.if_supports_color(Stream::Stdout, |t| t.magenta())
                                .to_string()
                        } else if Self::is_sort(&word) {
                            word.if_supports_color(Stream::Stdout, |t| t.cyan())
                                .to_string()
                        } else if Self::is_constant(&word) {
                            word.if_supports_color(Stream::Stdout, |t| t.yellow())
                                .to_string()
                        } else if word.chars().all(|c| c.is_ascii_digit() || c == '-') {
                            word.if_supports_color(Stream::Stdout, |t| t.yellow())
                                .to_string()
                        } else {
                            word.clone()
                        };
                        result.push_str(&colored);
                        word.clear();
                    }
                    result.push(ch);
                }
                _ => {
                    word.push(ch);
                }
            }
        }

        if !word.is_empty() {
            let colored = if Self::is_keyword(&word) {
                word.if_supports_color(Stream::Stdout, |t| t.blue())
                    .to_string()
            } else if Self::is_builtin_op(&word) {
                word.if_supports_color(Stream::Stdout, |t| t.magenta())
                    .to_string()
            } else if Self::is_sort(&word) {
                word.if_supports_color(Stream::Stdout, |t| t.cyan())
                    .to_string()
            } else if Self::is_constant(&word) {
                word.if_supports_color(Stream::Stdout, |t| t.yellow())
                    .to_string()
            } else if word.chars().all(|c| c.is_ascii_digit() || c == '-') {
                word.if_supports_color(Stream::Stdout, |t| t.yellow())
                    .to_string()
            } else {
                word
            };
            result.push_str(&colored);
        }

        std::borrow::Cow::Owned(result)
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        true
    }
}

impl Validator for SmtHighlighter {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        let input = ctx.input();
        let paren_count: i32 = input
            .chars()
            .map(|c| match c {
                '(' => 1,
                ')' => -1,
                _ => 0,
            })
            .sum();

        if paren_count > 0 {
            Ok(ValidationResult::Incomplete)
        } else {
            Ok(ValidationResult::Valid(None))
        }
    }
}

impl Completer for SmtHighlighter {
    type Candidate = Pair;

    fn complete(
        &self,
        _line: &str,
        _pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        Ok((0, vec![]))
    }
}

impl Hinter for SmtHighlighter {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        None
    }
}

impl Helper for SmtHighlighter {}

/// Run the interactive REPL mode
pub(crate) fn run_interactive(ctx: &mut Context, args: &Args, verbosity: Verbosity) {
    let config = Config::builder()
        .completion_type(CompletionType::List)
        .auto_add_history(true)
        .build();

    let highlighter = SmtHighlighter::new(!args.no_color);

    let mut rl: Editor<SmtHighlighter, FileHistory> =
        Editor::with_config(config).unwrap_or_else(|e| {
            eprintln_colored(args, &format!("Failed to initialize editor: {}", e));
            std::process::exit(1);
        });

    rl.set_helper(Some(highlighter));

    // Load history
    let history_path = dirs::home_dir()
        .map(|mut p| {
            p.push(".oxiz_history");
            p
        })
        .unwrap_or_else(|| PathBuf::from(".oxiz_history"));

    let _ = rl.load_history(&history_path);

    if verbosity >= Verbosity::Normal {
        println_colored(
            args,
            &format!("OxiZ SMT Solver v{}", env!("CARGO_PKG_VERSION")),
            Some(owo_colors::AnsiColors::Green),
        );
        println_colored(
            args,
            "Type (exit) to quit, (help) for help, or enter SMT-LIB2 commands.",
            None,
        );
        println!();
    }

    loop {
        let readline = rl.readline("oxiz> ");
        match readline {
            Ok(input) => {
                let input = input.trim();
                if input.is_empty() {
                    continue;
                }

                // Handle special commands
                if input == "(exit)" || input == "(quit)" {
                    break;
                }

                if input == "(help)" {
                    print_help(args);
                    continue;
                }

                if input == "(version)" {
                    print_version_info(args);
                    continue;
                }

                let start = Instant::now();
                let result = execute_and_format(ctx, input, args);
                let time_ms = start.elapsed().as_millis();

                if !result.is_empty() {
                    println!("{}", result);
                }

                if args.time && verbosity >= Verbosity::Verbose {
                    println_colored(
                        args,
                        &format!("Time: {}ms", time_ms),
                        Some(owo_colors::AnsiColors::BrightBlack),
                    );
                }
            }
            Err(ReadlineError::Interrupted) => {
                if verbosity >= Verbosity::Normal {
                    println_colored(args, "CTRL-C", None);
                }
                break;
            }
            Err(ReadlineError::Eof) => {
                if verbosity >= Verbosity::Normal {
                    println_colored(args, "CTRL-D", None);
                }
                break;
            }
            Err(err) => {
                eprintln_colored(args, &format!("Error: {}", err));
                break;
            }
        }
    }

    // Save history
    let _ = rl.save_history(&history_path);
}
