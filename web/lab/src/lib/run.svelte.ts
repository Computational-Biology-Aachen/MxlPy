// State of the run shared by all pages: the overview and the event stream.

import { api, ApiError, type Overview } from "./api.js";

class Run {
  overview = $state<Overview | null>(null);
  error = $state<string | null>(null);
  unauthorized = $state(false);
  /** Bumped when results, decisions or the run state change. */
  version = $state(0);
  /** Bumped (at most every 0.5 s) while fits report progress. */
  progress = $state(0);
  connected = $state(false);

  #refreshing: Promise<void> | null = null;

  async refresh(): Promise<void> {
    this.#refreshing ??= (async () => {
      try {
        this.overview = await api.overview();
        this.error = null;
        this.unauthorized = false;
      } catch (e) {
        this.unauthorized = e instanceof ApiError && e.status === 401;
        this.error = e instanceof Error ? e.message : String(e);
      } finally {
        this.#refreshing = null;
      }
    })();
    return this.#refreshing;
  }

  connect(): () => void {
    void this.refresh();
    const source = new EventSource("/api/events");
    let progressTimer: ReturnType<typeof setTimeout> | null = null;
    source.onopen = () => {
      this.connected = true;
      void this.refresh();
    };
    source.onerror = () => {
      this.connected = false;
    };
    source.onmessage = (msg) => {
      let ev: { source?: string };
      try {
        ev = JSON.parse(msg.data);
      } catch {
        return;
      }
      if (ev.source === "lab") {
        void this.refresh().then(() => this.version++);
      } else {
        progressTimer ??= setTimeout(() => {
          progressTimer = null;
          this.progress++;
        }, 500);
      }
    };
    return () => {
      source.close();
      if (progressTimer !== null) clearTimeout(progressTimer);
      this.connected = false;
    };
  }

  /** Run an action, then refresh; errors end up in `error`. */
  async act(fn: () => Promise<unknown>): Promise<boolean> {
    try {
      await fn();
      await this.refresh();
      this.version++;
      return true;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      return false;
    }
  }
}

export const run = new Run();
