


from tree_sitter import Language, Parser

Language.build_library(
  
  'my-languages.so',

  
  [
    'tree-sitter-python',
    'tree-sitter-java',
    'tree-sitter-cpp',
    'tree-sitter-c',

  ]
)

