/**
 * OxiZ SMT Solver VS Code Extension
 *
 * Provides language support for SMT-LIB2 files including:
 * - Syntax highlighting
 * - Diagnostics (errors/warnings)
 * - Code completion
 * - Hover information
 * - Document symbols
 * - Run solver command
 */

import * as vscode from "vscode";
import * as cp from "child_process";
import * as path from "path";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

// Output channel for solver results
let outputChannel: vscode.OutputChannel;

// LSP client instance
let client: LanguageClient | undefined;

// Diagnostics collection for non-LSP mode
let diagnosticsCollection: vscode.DiagnosticCollection;

/**
 * SMT-LIB2 keyword documentation for hover support
 */
const KEYWORD_DOCS: Record<string, string> = {
  "set-logic":
    "Sets the background logic for the SMT solver.\n\nSyntax: `(set-logic <symbol>)`\n\nExamples: QF_UF, QF_LIA, QF_BV, ALL",
  "declare-const":
    "Declares a constant with a given sort.\n\nSyntax: `(declare-const <symbol> <sort>)`\n\nExample: `(declare-const x Int)`",
  "declare-fun":
    "Declares a function with argument sorts and return sort.\n\nSyntax: `(declare-fun <symbol> (<sort>*) <sort>)`\n\nExample: `(declare-fun f (Int Int) Bool)`",
  "declare-sort":
    "Declares a new uninterpreted sort.\n\nSyntax: `(declare-sort <symbol> <numeral>)`",
  "define-fun":
    "Defines a function with a body.\n\nSyntax: `(define-fun <symbol> ((<symbol> <sort>)*) <sort> <term>)`\n\nExample: `(define-fun abs ((x Int)) Int (ite (>= x 0) x (- x)))`",
  "define-sort":
    "Defines a sort abbreviation.\n\nSyntax: `(define-sort <symbol> (<symbol>*) <sort>)`",
  assert:
    "Adds a formula to the current assertion stack.\n\nSyntax: `(assert <term>)`\n\nExample: `(assert (> x 0))`",
  "check-sat":
    "Checks satisfiability of the current assertions.\n\nReturns: `sat`, `unsat`, or `unknown`",
  "get-model":
    "Gets a satisfying model (when result is sat).\n\nReturns variable assignments.",
  "get-value":
    "Gets values of specified terms in the model.\n\nSyntax: `(get-value (<term>+))`",
  "get-proof": "Retrieves the proof (when result is unsat).",
  "get-unsat-core":
    "Gets the unsat core - minimal subset of assertions that are unsatisfiable.",
  push: "Creates a new assertion scope, saving the current state.\n\nSyntax: `(push)` or `(push <numeral>)`",
  pop: "Pops the assertion scope, restoring previous state.\n\nSyntax: `(pop)` or `(pop <numeral>)`",
  reset: "Resets the entire solver state to initial conditions.",
  exit: "Exits the SMT solver session.",
  echo: 'Prints a message.\n\nSyntax: `(echo "message")`',
  "set-info":
    "Sets solver metadata.\n\nSyntax: `(set-info :<keyword> <value>)`",
  "set-option":
    "Sets solver options.\n\nSyntax: `(set-option :<option> <value>)`",
  "get-info":
    "Retrieves solver information.\n\nSyntax: `(get-info :<keyword>)`",
  "get-option":
    "Gets the value of a solver option.\n\nSyntax: `(get-option :<option>)`",
  Int: "The sort of mathematical integers (arbitrary precision).",
  Real: "The sort of real numbers.",
  Bool: "The Boolean sort (true/false).",
  String: "The sort of strings.",
  Array:
    "Parametric sort for arrays.\n\nSyntax: `(Array <index-sort> <element-sort>)`",
  BitVec:
    "Parametric sort for bit-vectors.\n\nSyntax: `(_ BitVec <n>)` where n is the bit-width",
  true: "Boolean constant representing true.",
  false: "Boolean constant representing false.",
  and: "Logical AND operator. Returns true if all arguments are true.",
  or: "Logical OR operator. Returns true if any argument is true.",
  not: "Logical NOT operator. Negates a boolean value.",
  "=>": "Logical implication operator. `(=> a b)` means 'a implies b'.",
  "=": "Equality operator. Returns true if all arguments are equal.",
  distinct:
    "Pairwise distinct operator. Returns true if all arguments are pairwise different.",
  ite: "If-then-else operator.\n\nSyntax: `(ite <condition> <then> <else>)`",
  "+": "Addition operator for numeric types.",
  "-": "Subtraction operator for numeric types.",
  "*": "Multiplication operator for numeric types.",
  "/": "Division operator for real numbers.",
  div: "Integer division operator.",
  mod: "Modulo operator for integers.",
  "<": "Less-than comparison for numeric types.",
  ">": "Greater-than comparison for numeric types.",
  "<=": "Less-than-or-equal comparison for numeric types.",
  ">=": "Greater-than-or-equal comparison for numeric types.",
  let: "Local binding construct.\n\nSyntax: `(let ((<var> <value>) ...) <body>)`",
  forall:
    "Universal quantifier.\n\nSyntax: `(forall ((<var> <sort>) ...) <body>)`",
  exists:
    "Existential quantifier.\n\nSyntax: `(exists ((<var> <sort>) ...) <body>)`",
  select: "Array read operation.\n\nSyntax: `(select <array> <index>)`",
  store:
    "Array write operation.\n\nSyntax: `(store <array> <index> <value>)`",
  concat: "Bit-vector concatenation.",
  extract: "Bit-vector extraction.\n\nSyntax: `((_ extract <i> <j>) <bv>)`",
};

/**
 * SMT-LIB2 completion items
 */
function getCompletionItems(): vscode.CompletionItem[] {
  const items: vscode.CompletionItem[] = [];

  // Commands
  const commands = [
    {
      label: "set-logic",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Set the background logic",
      insertText: new vscode.SnippetString("set-logic ${1|ALL,QF_LIA,QF_BV,QF_UF,QF_LRA,QF_NIA|}"),
    },
    {
      label: "declare-const",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Declare a constant",
      insertText: new vscode.SnippetString("declare-const ${1:name} ${2|Int,Bool,Real|}"),
    },
    {
      label: "declare-fun",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Declare a function",
      insertText: new vscode.SnippetString("declare-fun ${1:name} (${2:}) ${3|Int,Bool,Real|}"),
    },
    {
      label: "define-fun",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Define a function",
      insertText: new vscode.SnippetString(
        "define-fun ${1:name} ((${2:arg} ${3:Int})) ${4:Int}\n  ${5:body}"
      ),
    },
    {
      label: "assert",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Add an assertion",
      insertText: new vscode.SnippetString("assert ${1:formula}"),
    },
    {
      label: "check-sat",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Check satisfiability",
      insertText: new vscode.SnippetString("check-sat"),
    },
    {
      label: "get-model",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Get the model",
      insertText: new vscode.SnippetString("get-model"),
    },
    {
      label: "get-value",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Get term values",
      insertText: new vscode.SnippetString("get-value (${1:terms})"),
    },
    {
      label: "push",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Push assertion scope",
      insertText: new vscode.SnippetString("push"),
    },
    {
      label: "pop",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Pop assertion scope",
      insertText: new vscode.SnippetString("pop"),
    },
    {
      label: "get-unsat-core",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Get unsat core",
      insertText: new vscode.SnippetString("get-unsat-core"),
    },
    {
      label: "get-proof",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Get proof",
      insertText: new vscode.SnippetString("get-proof"),
    },
    {
      label: "reset",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Reset solver state",
      insertText: new vscode.SnippetString("reset"),
    },
    {
      label: "exit",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Exit solver",
      insertText: new vscode.SnippetString("exit"),
    },
    {
      label: "set-option",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Set solver option",
      insertText: new vscode.SnippetString("set-option :${1:option} ${2:value}"),
    },
    {
      label: "set-info",
      kind: vscode.CompletionItemKind.Keyword,
      detail: "Set solver info",
      insertText: new vscode.SnippetString("set-info :${1:keyword} ${2:value}"),
    },
  ];

  for (const cmd of commands) {
    const item = new vscode.CompletionItem(cmd.label, cmd.kind);
    item.detail = cmd.detail;
    item.insertText = cmd.insertText;
    item.documentation = KEYWORD_DOCS[cmd.label];
    items.push(item);
  }

  // Types
  const types = ["Int", "Bool", "Real", "String", "Array", "BitVec"];
  for (const t of types) {
    const item = new vscode.CompletionItem(t, vscode.CompletionItemKind.TypeParameter);
    item.detail = `Sort: ${t}`;
    item.documentation = KEYWORD_DOCS[t];
    items.push(item);
  }

  // Operators
  const operators = [
    { label: "and", detail: "Logical AND" },
    { label: "or", detail: "Logical OR" },
    { label: "not", detail: "Logical NOT" },
    { label: "=>", detail: "Implication" },
    { label: "ite", detail: "If-then-else" },
    { label: "=", detail: "Equality" },
    { label: "distinct", detail: "Pairwise distinct" },
    { label: "+", detail: "Addition" },
    { label: "-", detail: "Subtraction" },
    { label: "*", detail: "Multiplication" },
    { label: "/", detail: "Division" },
    { label: "div", detail: "Integer division" },
    { label: "mod", detail: "Modulo" },
    { label: "<", detail: "Less than" },
    { label: ">", detail: "Greater than" },
    { label: "<=", detail: "Less than or equal" },
    { label: ">=", detail: "Greater than or equal" },
    { label: "let", detail: "Local binding" },
    { label: "forall", detail: "Universal quantifier" },
    { label: "exists", detail: "Existential quantifier" },
    { label: "select", detail: "Array read" },
    { label: "store", detail: "Array write" },
  ];

  for (const op of operators) {
    const item = new vscode.CompletionItem(op.label, vscode.CompletionItemKind.Operator);
    item.detail = op.detail;
    item.documentation = KEYWORD_DOCS[op.label];
    items.push(item);
  }

  // Logics
  const logics = [
    "ALL",
    "QF_UF",
    "QF_LIA",
    "QF_LRA",
    "QF_BV",
    "QF_NIA",
    "QF_NRA",
    "QF_AUFLIA",
    "QF_AUFBV",
    "HORN",
  ];
  for (const logic of logics) {
    const item = new vscode.CompletionItem(logic, vscode.CompletionItemKind.Constant);
    item.detail = `Logic: ${logic}`;
    items.push(item);
  }

  return items;
}

/**
 * Get configuration value
 */
function getConfig<T>(key: string, defaultValue: T): T {
  const config = vscode.workspace.getConfiguration("oxiz");
  return config.get<T>(key, defaultValue);
}

/**
 * Find oxiz executable path
 */
function findOxizPath(): string {
  const configPath = getConfig<string>("executablePath", "oxiz");

  // If it's an absolute path, use it directly
  if (path.isAbsolute(configPath)) {
    return configPath;
  }

  // Check workspace for local build
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (workspaceFolders) {
    for (const folder of workspaceFolders) {
      const localPath = path.join(
        folder.uri.fsPath,
        "target",
        "release",
        "oxiz"
      );
      try {
        cp.execSync(`test -x "${localPath}"`);
        return localPath;
      } catch {
        // Not found, continue
      }

      const debugPath = path.join(folder.uri.fsPath, "target", "debug", "oxiz");
      try {
        cp.execSync(`test -x "${debugPath}"`);
        return debugPath;
      } catch {
        // Not found, continue
      }
    }
  }

  // Return configured path (will be found in PATH)
  return configPath;
}

/**
 * Start the LSP client
 */
async function startLspClient(context: vscode.ExtensionContext): Promise<void> {
  const oxizPath = findOxizPath();

  const serverOptions: ServerOptions = {
    command: oxizPath,
    args: ["--lsp"],
    transport: TransportKind.stdio,
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "smtlib2" }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.smt2"),
    },
    outputChannel,
  };

  client = new LanguageClient(
    "oxiz-lsp",
    "OxiZ Language Server",
    serverOptions,
    clientOptions
  );

  try {
    await client.start();
    outputChannel.appendLine("OxiZ Language Server started");
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    outputChannel.appendLine(`Failed to start LSP server: ${errorMessage}`);
    vscode.window.showWarningMessage(
      `OxiZ LSP server failed to start: ${errorMessage}. Falling back to basic mode.`
    );
    client = undefined;
  }
}

/**
 * Stop the LSP client
 */
async function stopLspClient(): Promise<void> {
  if (client) {
    await client.stop();
    client = undefined;
  }
}

/**
 * Run the solver on the current file
 */
async function runSolver(document: vscode.TextDocument): Promise<void> {
  const oxizPath = findOxizPath();
  const timeout = getConfig<number>("timeout", 30);
  const logic = getConfig<string>("solver.logic", "");

  outputChannel.show(true);
  outputChannel.appendLine("----------------------------------------");
  outputChannel.appendLine(`Running OxiZ solver on: ${document.fileName}`);
  outputChannel.appendLine(`Time: ${new Date().toISOString()}`);
  outputChannel.appendLine("----------------------------------------");

  const args = ["--time"];

  if (logic) {
    args.push("--logic", logic);
  }

  if (timeout > 0) {
    args.push("--timeout", timeout.toString());
  }

  args.push(document.fileName);

  try {
    const result = await new Promise<string>((resolve, reject) => {
      const process = cp.spawn(oxizPath, args);
      let stdout = "";
      let stderr = "";

      process.stdout.on("data", (data: Buffer) => {
        stdout += data.toString();
      });

      process.stderr.on("data", (data: Buffer) => {
        stderr += data.toString();
      });

      process.on("close", (code) => {
        if (code === 0 || stdout.includes("sat") || stdout.includes("unsat")) {
          resolve(stdout);
        } else {
          reject(new Error(stderr || `Process exited with code ${code}`));
        }
      });

      process.on("error", (err) => {
        reject(err);
      });

      // Timeout handling
      if (timeout > 0) {
        setTimeout(() => {
          process.kill();
          reject(new Error("Solver timeout"));
        }, timeout * 1000);
      }
    });

    outputChannel.appendLine(result);
    outputChannel.appendLine("----------------------------------------");

    // Parse result for status bar
    if (result.includes("sat") && !result.includes("unsat")) {
      vscode.window.showInformationMessage("OxiZ: SAT (Satisfiable)");
    } else if (result.includes("unsat")) {
      vscode.window.showInformationMessage("OxiZ: UNSAT (Unsatisfiable)");
    } else if (result.includes("unknown")) {
      vscode.window.showWarningMessage("OxiZ: UNKNOWN");
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    outputChannel.appendLine(`Error: ${errorMessage}`);
    vscode.window.showErrorMessage(`OxiZ solver error: ${errorMessage}`);
  }
}

/**
 * Parse document and update diagnostics (fallback when LSP is not available)
 */
function updateDiagnostics(document: vscode.TextDocument): void {
  if (document.languageId !== "smtlib2") {
    return;
  }

  const diagnostics: vscode.Diagnostic[] = [];
  const text = document.getText();

  // Track parentheses
  const parenStack: { line: number; char: number }[] = [];

  for (let i = 0; i < document.lineCount; i++) {
    const line = document.lineAt(i);
    const lineText = line.text;

    // Skip comments
    const commentIndex = lineText.indexOf(";");
    const effectiveText =
      commentIndex >= 0 ? lineText.substring(0, commentIndex) : lineText;

    for (let j = 0; j < effectiveText.length; j++) {
      const char = effectiveText[j];
      if (char === "(") {
        parenStack.push({ line: i, char: j });
      } else if (char === ")") {
        if (parenStack.length === 0) {
          diagnostics.push(
            new vscode.Diagnostic(
              new vscode.Range(i, j, i, j + 1),
              "Unmatched closing parenthesis",
              vscode.DiagnosticSeverity.Error
            )
          );
        } else {
          parenStack.pop();
        }
      }
    }
  }

  // Report unclosed parentheses
  for (const pos of parenStack) {
    diagnostics.push(
      new vscode.Diagnostic(
        new vscode.Range(pos.line, pos.char, pos.line, pos.char + 1),
        "Unclosed parenthesis",
        vscode.DiagnosticSeverity.Error
      )
    );
  }

  // Check for common issues
  const lines = text.split("\n");
  let hasSetLogic = false;
  let hasCheckSat = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (line.includes("(set-logic")) {
      hasSetLogic = true;
    }
    if (line.includes("(check-sat")) {
      hasCheckSat = true;
    }
  }

  // Add hints
  if (!hasSetLogic && text.length > 10) {
    diagnostics.push(
      new vscode.Diagnostic(
        new vscode.Range(0, 0, 0, 0),
        "Consider adding (set-logic ...) at the beginning",
        vscode.DiagnosticSeverity.Hint
      )
    );
  }

  if (!hasCheckSat && text.includes("(assert")) {
    diagnostics.push(
      new vscode.Diagnostic(
        new vscode.Range(document.lineCount - 1, 0, document.lineCount - 1, 0),
        "Missing (check-sat) command",
        vscode.DiagnosticSeverity.Hint
      )
    );
  }

  diagnosticsCollection.set(document.uri, diagnostics);
}

/**
 * Extract document symbols from SMT-LIB2 file
 */
function getDocumentSymbols(
  document: vscode.TextDocument
): vscode.DocumentSymbol[] {
  const symbols: vscode.DocumentSymbol[] = [];
  const text = document.getText();
  const lines = text.split("\n");

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Match declare-const
    const declareConstMatch = trimmed.match(
      /^\(declare-const\s+([^\s)]+)\s+([^\s)]+)/
    );
    if (declareConstMatch) {
      const name = declareConstMatch[1];
      const sort = declareConstMatch[2];
      const range = new vscode.Range(i, 0, i, line.length);
      const symbol = new vscode.DocumentSymbol(
        name,
        sort,
        vscode.SymbolKind.Constant,
        range,
        range
      );
      symbols.push(symbol);
    }

    // Match declare-fun
    const declareFunMatch = trimmed.match(/^\(declare-fun\s+([^\s)]+)/);
    if (declareFunMatch) {
      const name = declareFunMatch[1];
      const range = new vscode.Range(i, 0, i, line.length);
      const symbol = new vscode.DocumentSymbol(
        name,
        "function",
        vscode.SymbolKind.Function,
        range,
        range
      );
      symbols.push(symbol);
    }

    // Match define-fun
    const defineFunMatch = trimmed.match(/^\(define-fun\s+([^\s)]+)/);
    if (defineFunMatch) {
      const name = defineFunMatch[1];
      const range = new vscode.Range(i, 0, i, line.length);
      const symbol = new vscode.DocumentSymbol(
        name,
        "function definition",
        vscode.SymbolKind.Function,
        range,
        range
      );
      symbols.push(symbol);
    }

    // Match assert (named)
    const assertMatch = trimmed.match(/^\(assert\s+\(!\s+[^:]+:named\s+([^\s)]+)/);
    if (assertMatch) {
      const name = assertMatch[1];
      const range = new vscode.Range(i, 0, i, line.length);
      const symbol = new vscode.DocumentSymbol(
        name,
        "assertion",
        vscode.SymbolKind.Property,
        range,
        range
      );
      symbols.push(symbol);
    }
  }

  return symbols;
}

/**
 * Extension activation
 */
export async function activate(
  context: vscode.ExtensionContext
): Promise<void> {
  outputChannel = vscode.window.createOutputChannel("OxiZ SMT Solver");
  diagnosticsCollection =
    vscode.languages.createDiagnosticCollection("smtlib2");

  outputChannel.appendLine("OxiZ SMT Solver extension activated");

  // Start LSP if enabled
  const lspEnabled = getConfig<boolean>("enableLsp", true);
  if (lspEnabled) {
    await startLspClient(context);
  }

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand("oxiz.runSolver", async () => {
      const editor = vscode.window.activeTextEditor;
      if (editor && editor.document.languageId === "smtlib2") {
        await runSolver(editor.document);
      } else {
        vscode.window.showWarningMessage("Please open an SMT-LIB2 file first");
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("oxiz.checkSat", async () => {
      const editor = vscode.window.activeTextEditor;
      if (editor && editor.document.languageId === "smtlib2") {
        await runSolver(editor.document);
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("oxiz.getModel", async () => {
      const editor = vscode.window.activeTextEditor;
      if (editor && editor.document.languageId === "smtlib2") {
        await runSolver(editor.document);
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("oxiz.restartLanguageServer", async () => {
      outputChannel.appendLine("Restarting language server...");
      await stopLspClient();
      if (getConfig<boolean>("enableLsp", true)) {
        await startLspClient(context);
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("oxiz.showSolverInfo", async () => {
      const oxizPath = findOxizPath();
      try {
        const result = cp.execSync(`${oxizPath} --version`).toString();
        vscode.window.showInformationMessage(`OxiZ Solver: ${result.trim()}`);
      } catch {
        vscode.window.showErrorMessage(
          "Could not get OxiZ solver information. Is it installed?"
        );
      }
    })
  );

  // Register hover provider (fallback when LSP not available)
  if (!client) {
    context.subscriptions.push(
      vscode.languages.registerHoverProvider("smtlib2", {
        provideHover(
          document: vscode.TextDocument,
          position: vscode.Position
        ): vscode.Hover | undefined {
          const range = document.getWordRangeAtPosition(
            position,
            /[a-zA-Z_][a-zA-Z0-9_\-]*|[+\-*/<>=]+/
          );
          if (!range) {
            return undefined;
          }

          const word = document.getText(range);
          const doc = KEYWORD_DOCS[word];

          if (doc) {
            return new vscode.Hover(new vscode.MarkdownString(doc), range);
          }

          return undefined;
        },
      })
    );
  }

  // Register completion provider (fallback when LSP not available)
  if (!client) {
    context.subscriptions.push(
      vscode.languages.registerCompletionItemProvider(
        "smtlib2",
        {
          provideCompletionItems(): vscode.CompletionItem[] {
            return getCompletionItems();
          },
        },
        "(",
        " "
      )
    );
  }

  // Register document symbol provider (fallback when LSP not available)
  if (!client) {
    context.subscriptions.push(
      vscode.languages.registerDocumentSymbolProvider("smtlib2", {
        provideDocumentSymbols(
          document: vscode.TextDocument
        ): vscode.DocumentSymbol[] {
          return getDocumentSymbols(document);
        },
      })
    );
  }

  // Register diagnostics (fallback when LSP not available)
  if (!client) {
    // Update diagnostics on document change
    context.subscriptions.push(
      vscode.workspace.onDidChangeTextDocument((event) => {
        if (event.document.languageId === "smtlib2") {
          const delay = getConfig<number>("diagnostics.delay", 500);
          setTimeout(() => updateDiagnostics(event.document), delay);
        }
      })
    );

    // Update diagnostics when document opens
    context.subscriptions.push(
      vscode.workspace.onDidOpenTextDocument((document) => {
        if (document.languageId === "smtlib2") {
          updateDiagnostics(document);
        }
      })
    );

    // Initial diagnostics for open documents
    for (const document of vscode.workspace.textDocuments) {
      if (document.languageId === "smtlib2") {
        updateDiagnostics(document);
      }
    }
  }

  // Configuration change handler
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration(async (event) => {
      if (event.affectsConfiguration("oxiz.enableLsp")) {
        const lspEnabled = getConfig<boolean>("enableLsp", true);
        if (lspEnabled && !client) {
          await startLspClient(context);
        } else if (!lspEnabled && client) {
          await stopLspClient();
        }
      }
    })
  );

  // Clean up
  context.subscriptions.push(outputChannel);
  context.subscriptions.push(diagnosticsCollection);
}

/**
 * Extension deactivation
 */
export async function deactivate(): Promise<void> {
  await stopLspClient();
}
