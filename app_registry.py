# app_registry.py

class ApplicationRegistry:
    """
    Tracks which applications are using which paths and manages notifications.
    """
    def __init__(self):
        # Maps router_path (as tuple) -> set of application references
        self.path_to_apps = {}

        # Maps application -> current path (as tuple)
        self.app_to_path = {}

    def register_path_usage(self, application, router_path):
        """
        Register that an application is using a specific path.

        Args:
            application: Application instance
            router_path (list): Router sequence the app is using
        """
        path_tuple = tuple(router_path)

        # Remove previous path registration for this app
        if application in self.app_to_path:
            old_path = self.app_to_path[application]
            if old_path in self.path_to_apps:
                self.path_to_apps[old_path].discard(application)
                if not self.path_to_apps[old_path]:
                    del self.path_to_apps[old_path]

        # Register new path
        if path_tuple not in self.path_to_apps:
            self.path_to_apps[path_tuple] = set()
        self.path_to_apps[path_tuple].add(application)
        self.app_to_path[application] = path_tuple

    def unregister_application(self, application):
        """
        Remove all registrations for an application.

        Args:
            application: Application instance to unregister
        """
        if application in self.app_to_path:
            path_tuple = self.app_to_path[application]
            if path_tuple in self.path_to_apps:
                self.path_to_apps[path_tuple].discard(application)
                if not self.path_to_apps[path_tuple]:
                    del self.path_to_apps[path_tuple]
            del self.app_to_path[application]

    def notify_path_down(self, router_path, affected_as_pairs):
        """
        Notify all applications using this path that it's down.

        Args:
            router_path (list): The failed path
            affected_as_pairs (list): List of (src_as, dst_as) tuples affected
        """
        path_tuple = tuple(router_path)
        apps_to_notify = self.path_to_apps.get(path_tuple, set()).copy()

        print(f"  Notifying {len(apps_to_notify)} application(s) using this path")

        for app in apps_to_notify:
            if hasattr(app, 'on_path_down'):
                app.on_path_down(router_path)

    def notify_path_up(self, router_path, affected_as_pairs):
        """
        Notify applications that a path has recovered.

        Args:
            router_path (list): The recovered path
            affected_as_pairs (list): List of (src_as, dst_as) tuples affected
        """
        path_tuple = tuple(router_path)
        apps_to_notify = self.path_to_apps.get(path_tuple, set()).copy()

        print(f"  Notifying {len(apps_to_notify)} application(s) about path recovery")

        for app in apps_to_notify:
            if hasattr(app, 'on_path_up'):
                app.on_path_up(router_path)

    def get_apps_using_path(self, router_path):
        """
        Get set of applications using a specific path.

        Args:
            router_path (list): Router sequence

        Returns:
            set: Set of applications using this path
        """
        path_tuple = tuple(router_path)
        return self.path_to_apps.get(path_tuple, set()).copy()
