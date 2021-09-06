


from tree_sitter import Language, Parser

Language.build_library(
  
  'my-languages.so',

  
  [
    'tree-sitter-go',
    'tree-sitter-javascript',
    'tree-sitter-python',
    'tree-sitter-php',
    'tree-sitter-java',
    'tree-sitter-ruby',
    'tree-sitter-c-sharp',
  ]
)

