# Hierarchical Action Batching - Architecture Documentation

This document provides comprehensive architecture diagrams for the Hierarchical Action Batching system implemented in the AGI Agents framework.

## Overview

The system implements an adaptive web automation agent with plan-based execution and dynamic error recovery. It uses parallel analysis from Qwen VLM and GPT-4o to create and execute action plans efficiently.

## Architecture Diagrams

### 1. System Architecture (`architecture.puml`)

**Purpose:** Shows the complete system architecture with all components, their relationships, and data models.

**Key Components:**
- **AdaptiveAgent**: Main orchestration component that coordinates all other components
- **QwenVisionModel**: Vision model for page analysis and element location
- **GPT4Orchestrator**: Strategic decision-maker for plan creation and revision
- **ActionBatchExecutor**: Executes batches of actions efficiently
- **MetricsTracker**: Tracks performance metrics throughout execution

**Key Features:**
- Parallel analysis architecture (Vision + Orchestrator)
- Dynamic plan revision on errors
- Batched action execution for performance
- Comprehensive metrics tracking

### 2. Execution Flow (`execution-flow.puml`)

**Purpose:** Sequence diagram showing the complete execution flow from start to finish.

**Flow Steps:**
1. **Initialization**: Start metrics tracking
2. **Screenshot Capture**: Capture current page state
3. **Parallel Analysis**: Vision and Orchestrator run simultaneously
4. **State Update**: Update agent state with analysis results
5. **Action Batch Execution**: Execute consecutive same-page actions
6. **Result Handling**: Process success or trigger error recovery
7. **Return State**: Return updated state to caller

**Special Cases:**
- Conditional screenshot capture (only on navigation)
- Partial batch success handling
- Error recovery with plan revision

### 3. Error Recovery Flow (`error-recovery-flow.puml`)

**Purpose:** Activity diagram showing the error recovery and plan revision process.

**Recovery Process:**
1. Detect execution failure
2. Capture error screenshot
3. Check revision limit
4. Call Vision Model for updated analysis
5. Call Orchestrator with error information
6. Generate revised plan
7. Continue execution with new plan

**Revision Strategies:**
- Use different elements
- Change action sequence
- Add intermediate steps
- Adjust parameters

### 4. Component Interactions (`component-interactions.puml`)

**Purpose:** Shows how components interact and what data flows between them.

**Data Flow:**
1. Agent → Vision: Request page analysis
2. Agent → Orchestrator: Request execution plan
3. Agent → State: Store current analysis and plan
4. Agent → Executor: Create and execute action batch
5. Agent → Metrics: Record all events
6. Agent → Browser: Capture screenshots and execute actions

## Viewing the Diagrams

### Option 1: PlantUML Online Server
Visit [PlantUML Online](http://www.plantuml.com/plantuml/uml/) and paste the contents of any `.puml` file.

### Option 2: VS Code Extension
1. Install the "PlantUML" extension by jebbs
2. Open any `.puml` file
3. Press `Alt+D` to preview

### Option 3: Command Line
```bash
# Install PlantUML
brew install plantuml  # macOS
# or
sudo apt-get install plantuml  # Linux

# Generate PNG images
plantuml architecture.puml
plantuml execution-flow.puml
plantuml error-recovery-flow.puml
plantuml component-interactions.puml
```

### Option 4: Docker
```bash
docker run -v $(pwd):/data plantuml/plantuml:latest \
    architecture.puml \
    execution-flow.puml \
    error-recovery-flow.puml \
    component-interactions.puml
```

## Key Design Patterns

### 1. Parallel Processing
- Vision and Orchestrator models run simultaneously using `asyncio.gather()`
- Reduces latency by overlapping I/O operations
- Improves overall system throughput

### 2. Batch Execution
- Groups consecutive same-page actions together
- Executes without intermediate screenshots
- Stops on first error or navigation
- Significantly reduces screenshot overhead

### 3. Dynamic Plan Revision
- Plans are revised when errors occur
- Orchestrator receives detailed error context
- Supports partial batch success recovery
- Limits consecutive revisions to prevent infinite loops

### 4. Turn 1 Constraints
- First turn limited to exploratory actions only
- Prevents assumptions about page structure
- Ensures proper page observation before detailed planning
- Validated automatically by orchestrator

### 5. Metrics Tracking
- Comprehensive performance monitoring
- Tracks all model invocations
- Measures action execution success rates
- Calculates efficiency metrics (actions per screenshot)

## Data Models

### Vision Model Outputs
- **PageElement**: Individual interactive element
- **PageAnalysis**: Complete page analysis with all elements
- **ElementLocation**: Successful element location result
- **ElementLocationFailure**: Failed location with available alternatives

### Orchestrator Outputs
- **Action**: Single executable action
- **ExecutionPlan**: Sequence of actions with reasoning
- **ToolCall**: Direct action execution request
- **InfoSeekingQuery**: Element location request

### Execution Results
- **ExecutionResult**: Batch execution outcome
- **ExecutionError**: Detailed error information
- **MetricsSummary**: Performance statistics

## Performance Optimizations

1. **Conditional Screenshot Capture**: Only capture after navigation
2. **Parallel Model Calls**: Vision and Orchestrator run simultaneously
3. **Batch Execution**: Multiple actions per screenshot
4. **Smart Orchestrator Skipping**: Skip when plan has valid actions
5. **Partial Success Recovery**: Preserve progress on batch failures

## Error Handling

### Error Types
- `element_not_found`: Target element not on page
- `type_mismatch`: Action incompatible with element type
- `timeout`: Action exceeded time limit
- `navigation_error`: Navigation failed
- `unknown_error`: Unclassified error

### Recovery Strategies
1. Capture error context (screenshot, error details)
2. Update page analysis with current state
3. Provide error information to orchestrator
4. Generate revised plan with alternative approach
5. Continue execution with new plan

### Revision Limits
- Maximum 3 consecutive revisions
- Prevents infinite retry loops
- Fails task gracefully after limit
- Provides detailed failure information

## Integration Points

### Browser Integration
- Uses `AgentBrowser` for all browser interactions
- Supports screenshot capture with retry logic
- Executes actions via mouse/keyboard primitives
- Handles page navigation and waiting

### State Management
- Updates `AgentState` with execution progress
- Maintains message history for context
- Tracks step counter and completion status
- Stores task goal and current URL

### Metrics Integration
- Records all significant events
- Provides real-time performance tracking
- Generates comprehensive summary reports
- Supports performance analysis and optimization

## Future Enhancements

1. **Multi-page Planning**: Support for cross-page action sequences
2. **Adaptive Batching**: Dynamic batch size based on action types
3. **Predictive Caching**: Cache common page analyses
4. **Smart Retry**: Exponential backoff for transient errors
5. **Parallel Action Execution**: Execute independent actions simultaneously

## References

- Design Document: `.kiro/specs/hierarchical-action-batching/design.md`
- Requirements: `.kiro/specs/hierarchical-action-batching/requirements.md`
- Tasks: `.kiro/specs/hierarchical-action-batching/tasks.md`
- Implementation: `src/agi_agents/`
- Tests: `tests/`
