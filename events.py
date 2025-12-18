# events.py

class EventManager:
    """
    Manages scheduled network events (path failures, recoveries) in the simulation.
    """
    def __init__(self, env, path_selection_algorithm, application_registry):
        self.env = env
        self.path_selector = path_selection_algorithm
        self.app_registry = application_registry
        self.events = []

    def load_events(self, config_dict):
        """
        Load events from configuration dictionary.

        Args:
            config_dict (dict): Configuration with 'events' key containing event list
        """
        if 'events' not in config_dict:
            return

        events = config_dict['events']

        for event in events:
            # Validate required fields
            if 'type' not in event:
                print(f"Warning: Event missing 'type' field: {event}")
                continue
            if 'time_ms' not in event:
                print(f"Warning: Event missing 'time_ms' field: {event}")
                continue
            if 'path' not in event:
                print(f"Warning: Event missing 'path' field: {event}")
                continue

            # Validate event type
            if event['type'] not in ['path_down', 'path_up']:
                print(f"Warning: Unknown event type '{event['type']}': {event}")
                continue

            self.events.append(event)

        # Sort events by time
        self.events.sort(key=lambda e: e['time_ms'])
        print(f"Loaded {len(self.events)} event(s) from configuration")

    def schedule_events(self):
        """
        Schedule all loaded events in the SimPy environment.
        This is a SimPy generator process.
        """
        if not self.events:
            return

        print(f"Event scheduler started with {len(self.events)} event(s)")

        for event in self.events:
            # Wait until event time
            current_time = self.env.now
            event_time = event['time_ms']

            if event_time > current_time:
                yield self.env.timeout(event_time - current_time)

            # Execute event based on type
            if event['type'] == 'path_down':
                self._execute_path_down(event)
            elif event['type'] == 'path_up':
                self._execute_path_up(event)

    def _execute_path_down(self, event):
        """
        Execute a path_down event.

        Args:
            event (dict): Event configuration with 'path' and optional 'description'
        """
        router_path = event['path']
        description = event.get('description', '')

        print(f"[{self.env.now:.2f}] EVENT: Path down - {router_path}")
        if description:
            print(f"[{self.env.now:.2f}]   Description: {description}")

        # Mark path as down in path selector
        affected_pairs = self.path_selector.mark_path_down(router_path)

        print(f"[{self.env.now:.2f}]   Affected AS pairs: {affected_pairs}")

        # Notify applications
        self.app_registry.notify_path_down(router_path, affected_pairs)

    def _execute_path_up(self, event):
        """
        Execute a path_up event.

        Args:
            event (dict): Event configuration with 'path' and optional 'description'
        """
        router_path = event['path']
        description = event.get('description', '')

        print(f"[{self.env.now:.2f}] EVENT: Path up - {router_path}")
        if description:
            print(f"[{self.env.now:.2f}]   Description: {description}")

        # Mark path as up in path selector
        affected_pairs = self.path_selector.mark_path_up(router_path)

        print(f"[{self.env.now:.2f}]   Affected AS pairs: {affected_pairs}")

        # Optionally notify applications about recovery
        self.app_registry.notify_path_up(router_path, affected_pairs)
