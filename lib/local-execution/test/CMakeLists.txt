ff_add_test_executable(
  NAME
    local-execution-tests
  SRC_PATTERNS
    src/*.cc
  PRIVATE_INCLUDE 
    src/
  DEPS
    doctest
    utils-test-common
    local-execution
    kernels
    op-attrs
    task-spec
)

