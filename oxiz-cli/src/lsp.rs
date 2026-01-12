//! LSP Server implementation for OxiZ SMT Solver

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

/// Stores information about declared symbols in a document
#[derive(Debug, Clone)]
struct DocumentSymbols {
    /// Declared constants and functions
    symbols: HashMap<String, SymbolInfo>,
}

#[derive(Debug, Clone)]
struct SymbolInfo {
    name: String,
    kind: SymbolKind,
    range: Range,
}

/// LSP backend for OxiZ SMT Solver
pub struct OxizBackend {
    client: Client,
    document_map: Arc<RwLock<HashMap<Url, String>>>,
    symbol_map: Arc<RwLock<HashMap<Url, DocumentSymbols>>>,
}

impl OxizBackend {
    /// Create a new LSP backend
    pub fn new(client: Client) -> Self {
        Self {
            client,
            document_map: Arc::new(RwLock::new(HashMap::new())),
            symbol_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Extract symbols from SMT-LIB2 document
    fn extract_symbols(&self, text: &str) -> DocumentSymbols {
        let mut symbols = HashMap::new();
        let lines: Vec<&str> = text.lines().collect();

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Match declare-const: (declare-const <name> <sort>)
            if trimmed.starts_with("(declare-const ")
                && let Some(name_start) = trimmed.find("declare-const ")
            {
                let after_keyword = &trimmed[name_start + 14..];
                if let Some(name_end) = after_keyword.find(char::is_whitespace) {
                    let name = after_keyword[..name_end].trim().to_string();
                    let start_col = line.find(&name).unwrap_or(0) as u32;

                    symbols.insert(
                        name.clone(),
                        SymbolInfo {
                            name,
                            kind: SymbolKind::CONSTANT,
                            range: Range {
                                start: Position {
                                    line: line_num as u32,
                                    character: start_col,
                                },
                                end: Position {
                                    line: line_num as u32,
                                    character: start_col + name_end as u32,
                                },
                            },
                        },
                    );
                }
            }

            // Match declare-fun: (declare-fun <name> (<args>) <return>)
            if trimmed.starts_with("(declare-fun ")
                && let Some(name_start) = trimmed.find("declare-fun ")
            {
                let after_keyword = &trimmed[name_start + 12..];
                if let Some(name_end) = after_keyword.find(char::is_whitespace) {
                    let name = after_keyword[..name_end].trim().to_string();
                    let start_col = line.find(&name).unwrap_or(0) as u32;

                    symbols.insert(
                        name.clone(),
                        SymbolInfo {
                            name,
                            kind: SymbolKind::FUNCTION,
                            range: Range {
                                start: Position {
                                    line: line_num as u32,
                                    character: start_col,
                                },
                                end: Position {
                                    line: line_num as u32,
                                    character: start_col + name_end as u32,
                                },
                            },
                        },
                    );
                }
            }

            // Match define-fun: (define-fun <name> ...)
            if trimmed.starts_with("(define-fun ")
                && let Some(name_start) = trimmed.find("define-fun ")
            {
                let after_keyword = &trimmed[name_start + 11..];
                if let Some(name_end) = after_keyword.find(char::is_whitespace) {
                    let name = after_keyword[..name_end].trim().to_string();
                    let start_col = line.find(&name).unwrap_or(0) as u32;

                    symbols.insert(
                        name.clone(),
                        SymbolInfo {
                            name,
                            kind: SymbolKind::FUNCTION,
                            range: Range {
                                start: Position {
                                    line: line_num as u32,
                                    character: start_col,
                                },
                                end: Position {
                                    line: line_num as u32,
                                    character: start_col + name_end as u32,
                                },
                            },
                        },
                    );
                }
            }
        }

        DocumentSymbols { symbols }
    }

    /// Validate SMT-LIB2 document and return diagnostics
    async fn validate_document(&self, _uri: &Url, text: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let mut paren_stack = Vec::new();
        let mut line = 0u32;
        let mut col = 0u32;

        for (idx, ch) in text.chars().enumerate() {
            match ch {
                '(' => {
                    paren_stack.push((line, col, idx));
                }
                ')' => {
                    if paren_stack.is_empty() {
                        diagnostics.push(Diagnostic {
                            range: Range {
                                start: Position {
                                    line,
                                    character: col,
                                },
                                end: Position {
                                    line,
                                    character: col + 1,
                                },
                            },
                            severity: Some(DiagnosticSeverity::ERROR),
                            code: None,
                            source: Some("oxiz-lsp".to_string()),
                            message: "Unmatched closing parenthesis".to_string(),
                            related_information: None,
                            tags: None,
                            code_description: None,
                            data: None,
                        });
                    } else {
                        paren_stack.pop();
                    }
                }
                '\n' => {
                    line += 1;
                    col = 0;
                    continue;
                }
                _ => {}
            }
            col += 1;
        }

        // Check for unclosed parentheses
        for (line, col, _) in paren_stack {
            diagnostics.push(Diagnostic {
                range: Range {
                    start: Position {
                        line,
                        character: col,
                    },
                    end: Position {
                        line,
                        character: col + 1,
                    },
                },
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                source: Some("oxiz-lsp".to_string()),
                message: "Unclosed parenthesis".to_string(),
                related_information: None,
                tags: None,
                code_description: None,
                data: None,
            });
        }

        diagnostics
    }

    /// Get hover information for SMT-LIB2 keywords
    #[allow(dead_code)]
    fn get_keyword_hover(keyword: &str) -> Option<String> {
        match keyword {
            "set-logic" => Some("Sets the background logic for the SMT solver. Examples: QF_UF, QF_LIA, QF_BV, ALL".to_string()),
            "declare-const" => Some("Declares a constant with a given sort. Syntax: (declare-const <name> <sort>)".to_string()),
            "declare-fun" => Some("Declares a function with argument sorts and return sort. Syntax: (declare-fun <name> (<arg-sorts>) <return-sort>)".to_string()),
            "declare-sort" => Some("Declares a new uninterpreted sort. Syntax: (declare-sort <name> <arity>)".to_string()),
            "define-fun" => Some("Defines a function with a body. Syntax: (define-fun <name> (<params>) <sort> <body>)".to_string()),
            "define-sort" => Some("Defines a sort abbreviation. Syntax: (define-sort <name> (<params>) <sort>)".to_string()),
            "assert" => Some("Adds a formula to the current assertion stack. Syntax: (assert <formula>)".to_string()),
            "check-sat" => Some("Checks satisfiability of the current assertions. Returns: sat, unsat, or unknown".to_string()),
            "get-model" => Some("Gets a satisfying model (when result is sat). Returns variable assignments.".to_string()),
            "get-value" => Some("Gets values of specified terms in the model. Syntax: (get-value (<term1> <term2> ...))".to_string()),
            "get-proof" => Some("Retrieves the proof (when result is unsat).".to_string()),
            "get-unsat-core" => Some("Gets the unsat core - minimal subset of assertions that are unsatisfiable.".to_string()),
            "push" => Some("Creates a new assertion scope, saving the current state. Syntax: (push) or (push <n>)".to_string()),
            "pop" => Some("Pops the assertion scope, restoring previous state. Syntax: (pop) or (pop <n>)".to_string()),
            "reset" => Some("Resets the entire solver state to initial conditions.".to_string()),
            "exit" => Some("Exits the SMT solver session.".to_string()),
            "echo" => Some("Prints a message. Syntax: (echo \"message\")".to_string()),
            "set-info" => Some("Sets solver metadata. Syntax: (set-info :<keyword> <value>)".to_string()),
            "set-option" => Some("Sets solver options. Syntax: (set-option :<option> <value>)".to_string()),
            "get-info" => Some("Retrieves solver information. Syntax: (get-info :<keyword>)".to_string()),
            "get-option" => Some("Gets the value of a solver option. Syntax: (get-option :<option>)".to_string()),
            "Int" => Some("The sort of mathematical integers (arbitrary precision).".to_string()),
            "Real" => Some("The sort of real numbers.".to_string()),
            "Bool" => Some("The Boolean sort (true/false).".to_string()),
            "String" => Some("The sort of strings.".to_string()),
            "Array" => Some("Parametric sort for arrays. Syntax: (Array <index-sort> <element-sort>)".to_string()),
            "BitVec" => Some("Parametric sort for bit-vectors. Syntax: (_ BitVec <n>) where n is the bit-width".to_string()),
            "true" => Some("Boolean constant representing true.".to_string()),
            "false" => Some("Boolean constant representing false.".to_string()),
            "and" => Some("Logical AND operator. Returns true if all arguments are true.".to_string()),
            "or" => Some("Logical OR operator. Returns true if any argument is true.".to_string()),
            "not" => Some("Logical NOT operator. Negates a boolean value.".to_string()),
            "=>" => Some("Logical implication operator. (=> a b) means 'a implies b'.".to_string()),
            "=" => Some("Equality operator. Returns true if all arguments are equal.".to_string()),
            "distinct" => Some("Pairwise distinct operator. Returns true if all arguments are pairwise different.".to_string()),
            "ite" => Some("If-then-else operator. Syntax: (ite <condition> <then> <else>)".to_string()),
            "+" => Some("Addition operator for numeric types.".to_string()),
            "-" => Some("Subtraction operator for numeric types.".to_string()),
            "*" => Some("Multiplication operator for numeric types.".to_string()),
            "/" => Some("Division operator for real numbers.".to_string()),
            "div" => Some("Integer division operator.".to_string()),
            "mod" => Some("Modulo operator for integers.".to_string()),
            "<" => Some("Less-than comparison for numeric types.".to_string()),
            ">" => Some("Greater-than comparison for numeric types.".to_string()),
            "<=" => Some("Less-than-or-equal comparison for numeric types.".to_string()),
            ">=" => Some("Greater-than-or-equal comparison for numeric types.".to_string()),
            "let" => Some("Local binding construct. Syntax: (let ((<var> <value>) ...) <body>)".to_string()),
            "forall" => Some("Universal quantifier. Syntax: (forall ((<var> <sort>) ...) <body>)".to_string()),
            "exists" => Some("Existential quantifier. Syntax: (exists ((<var> <sort>) ...) <body>)".to_string()),
            _ => None,
        }
    }

    /// Get completion items for SMT-LIB2
    #[allow(dead_code)]
    fn get_completions() -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "set-logic".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Set the background logic".to_string()),
                documentation: Some(Documentation::String(
                    "Sets the background logic for the SMT solver.".to_string(),
                )),
                insert_text: Some("set-logic ".to_string()),
                ..Default::default()
            },
            CompletionItem {
                label: "declare-const".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Declare a constant".to_string()),
                documentation: Some(Documentation::String(
                    "Declares a constant with a given sort.".to_string(),
                )),
                insert_text: Some("declare-const ${1:name} ${2:Int}".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                ..Default::default()
            },
            CompletionItem {
                label: "declare-fun".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Declare a function".to_string()),
                documentation: Some(Documentation::String(
                    "Declares a function with argument sorts and return sort.".to_string(),
                )),
                insert_text: Some("declare-fun ${1:name} (${2:}) ${3:Int}".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                ..Default::default()
            },
            CompletionItem {
                label: "assert".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Add an assertion".to_string()),
                documentation: Some(Documentation::String(
                    "Adds a formula to the current assertion stack.".to_string(),
                )),
                insert_text: Some("assert ${1:formula}".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                ..Default::default()
            },
            CompletionItem {
                label: "check-sat".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Check satisfiability".to_string()),
                documentation: Some(Documentation::String(
                    "Checks satisfiability of the current assertions.".to_string(),
                )),
                ..Default::default()
            },
            CompletionItem {
                label: "get-model".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Get the model".to_string()),
                documentation: Some(Documentation::String(
                    "Gets a satisfying model (when result is sat).".to_string(),
                )),
                ..Default::default()
            },
            CompletionItem {
                label: "get-value".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Get term values".to_string()),
                documentation: Some(Documentation::String(
                    "Gets values of specified terms in the model.".to_string(),
                )),
                insert_text: Some("get-value (${1:terms})".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                ..Default::default()
            },
            CompletionItem {
                label: "push".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Push assertion scope".to_string()),
                documentation: Some(Documentation::String(
                    "Creates a new assertion scope, saving the current state.".to_string(),
                )),
                ..Default::default()
            },
            CompletionItem {
                label: "pop".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Pop assertion scope".to_string()),
                documentation: Some(Documentation::String(
                    "Pops the assertion scope, restoring previous state.".to_string(),
                )),
                ..Default::default()
            },
            CompletionItem {
                label: "Int".to_string(),
                kind: Some(CompletionItemKind::TYPE_PARAMETER),
                detail: Some("Integer sort".to_string()),
                documentation: Some(Documentation::String(
                    "The sort of mathematical integers.".to_string(),
                )),
                ..Default::default()
            },
            CompletionItem {
                label: "Bool".to_string(),
                kind: Some(CompletionItemKind::TYPE_PARAMETER),
                detail: Some("Boolean sort".to_string()),
                documentation: Some(Documentation::String(
                    "The Boolean sort (true/false).".to_string(),
                )),
                ..Default::default()
            },
            CompletionItem {
                label: "Real".to_string(),
                kind: Some(CompletionItemKind::TYPE_PARAMETER),
                detail: Some("Real number sort".to_string()),
                documentation: Some(Documentation::String(
                    "The sort of real numbers.".to_string(),
                )),
                ..Default::default()
            },
        ]
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for OxizBackend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec!["(".to_string(), " ".to_string()]),
                    ..Default::default()
                }),
                diagnostic_provider: Some(DiagnosticServerCapabilities::Options(
                    DiagnosticOptions {
                        identifier: Some("oxiz-lsp".to_string()),
                        inter_file_dependencies: false,
                        workspace_diagnostics: false,
                        work_done_progress_options: WorkDoneProgressOptions {
                            work_done_progress: None,
                        },
                    },
                )),
                document_symbol_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "oxiz-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "OxiZ LSP server initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let text = params.text_document.text;

        // Store document
        self.document_map
            .write()
            .await
            .insert(uri.clone(), text.clone());

        // Extract symbols
        let symbols = self.extract_symbols(&text);
        self.symbol_map.write().await.insert(uri.clone(), symbols);

        // Validate and send diagnostics
        let diagnostics = self.validate_document(&uri, &text).await;
        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;

        if let Some(change) = params.content_changes.first() {
            let text = change.text.clone();

            // Update document
            self.document_map
                .write()
                .await
                .insert(uri.clone(), text.clone());

            // Extract symbols
            let symbols = self.extract_symbols(&text);
            self.symbol_map.write().await.insert(uri.clone(), symbols);

            // Validate and send diagnostics
            let diagnostics = self.validate_document(&uri, &text).await;
            self.client
                .publish_diagnostics(uri, diagnostics, None)
                .await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        // Remove document and symbols from map
        self.document_map
            .write()
            .await
            .remove(&params.text_document.uri);
        self.symbol_map
            .write()
            .await
            .remove(&params.text_document.uri);
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let documents = self.document_map.read().await;
        if let Some(text) = documents.get(&uri) {
            // Extract word at position
            let lines: Vec<&str> = text.lines().collect();
            if let Some(line) = lines.get(position.line as usize) {
                let chars: Vec<char> = line.chars().collect();
                let pos = position.character as usize;

                if pos < chars.len() {
                    // Find word boundaries
                    let mut start = pos;
                    let mut end = pos;

                    // Move start backward
                    while start > 0 && !matches!(chars[start - 1], '(' | ')' | ' ' | '\t' | '\n') {
                        start -= 1;
                    }

                    // Move end forward
                    while end < chars.len() && !matches!(chars[end], '(' | ')' | ' ' | '\t' | '\n')
                    {
                        end += 1;
                    }

                    let word: String = chars[start..end].iter().collect();

                    if let Some(hover_text) = Self::get_keyword_hover(&word) {
                        return Ok(Some(Hover {
                            contents: HoverContents::Scalar(MarkedString::String(hover_text)),
                            range: Some(Range {
                                start: Position {
                                    line: position.line,
                                    character: start as u32,
                                },
                                end: Position {
                                    line: position.line,
                                    character: end as u32,
                                },
                            }),
                        }));
                    }
                }
            }
        }

        Ok(None)
    }

    async fn completion(&self, _: CompletionParams) -> Result<Option<CompletionResponse>> {
        Ok(Some(CompletionResponse::Array(Self::get_completions())))
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = params.text_document.uri;
        let symbols_map = self.symbol_map.read().await;

        if let Some(doc_symbols) = symbols_map.get(&uri) {
            let mut response_symbols = Vec::new();

            for sym_info in doc_symbols.symbols.values() {
                #[allow(deprecated)]
                response_symbols.push(DocumentSymbol {
                    name: sym_info.name.clone(),
                    detail: None,
                    kind: sym_info.kind,
                    tags: None,
                    deprecated: None,
                    range: sym_info.range,
                    selection_range: sym_info.range,
                    children: None,
                });
            }

            return Ok(Some(DocumentSymbolResponse::Nested(response_symbols)));
        }

        Ok(None)
    }
}

/// Run the LSP server
pub async fn run_lsp_server() -> anyhow::Result<()> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(OxizBackend::new);
    Server::new(stdin, stdout, socket).serve(service).await;

    Ok(())
}
