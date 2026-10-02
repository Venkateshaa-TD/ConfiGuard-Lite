import { Component, type ErrorInfo, type ReactNode } from "react";

interface State { failed: boolean }

/** Catches rendering errors (e.g. an unexpected response shape) so the page never goes blank.
 *  Nothing about the error is shown or logged beyond a generic message. */
export class ErrorBoundary extends Component<{ children: ReactNode; onReset?: () => void }, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(_error: Error, _info: ErrorInfo): void {
    // Intentionally silent: no media, filenames or response content are logged.
  }

  reset = () => {
    this.setState({ failed: false });
    this.props.onReset?.();
  };

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div role="alert" className="rounded-lg border border-fake bg-fake-soft p-4 text-sm text-ink">
        <p className="font-semibold text-fake">The result could not be displayed.</p>
        <p className="mt-1 text-ink-2">No verdict is shown. Please analyse the file again.</p>
        <button type="button" onClick={this.reset}
          className="tactile mt-3 min-h-11 rounded-lg border border-line-strong bg-surface px-4 text-sm font-medium">
          Reset
        </button>
      </div>
    );
  }
}
