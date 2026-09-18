from prometheus_client import REGISTRY, Counter

SHARE_CREATED = Counter(
    name="share_created_total",
    documentation="Total number of shares successfully created",
    registry=REGISTRY,
)

SHARE_ACCESSED = Counter(
    name="share_accessed_total",
    documentation="Total number of shares successfully retrieved",
    registry=REGISTRY,
)

SHARE_NOT_FOUND = Counter(
    name="share_not_found_total",
    documentation="Total number of share retrievals returning 404 (expired or invalid)",
    registry=REGISTRY,
)

SHARE_DELETED = Counter(
    name="share_deleted_total",
    documentation="Total number of shares successfully deleted",
    registry=REGISTRY,
)

SHARE_DELETE_NOT_FOUND = Counter(
    name="share_delete_not_found_total",
    documentation="Total number of share deletions returning 404 (unknown or already-used deletion_id)",
    registry=REGISTRY,
)
