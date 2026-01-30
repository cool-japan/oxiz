//! Property-based tests for Datalog engine
//!
//! Tests:
//! - Schema creation
//! - Relation operations

use oxiz_core::datalog::{Relation, RelationId, RelationKind, Schema};
use proptest::prelude::*;

proptest! {
    /// Test that schema can be created with a name
    #[test]
    fn schema_creation(name in "[a-z][a-z0-9]{0,5}") {
        let schema = Schema::new(name.clone());
        prop_assert_eq!(schema.name(), &name);
    }

    /// Test that schema has zero columns initially
    #[test]
    fn schema_initial_columns(name in "[a-z][a-z0-9]{0,5}") {
        let schema = Schema::new(name);
        prop_assert_eq!(schema.arity(), 0);
    }

    /// Test that relation can be created
    #[test]
    fn relation_creation(name in "[a-z][a-z0-9]{0,5}") {
        let schema = Schema::new(name.clone());
        let rel_id = RelationId::new(1);
        let relation = Relation::new(rel_id, name.clone(), schema, RelationKind::Edb);
        prop_assert_eq!(relation.name(), &name);
    }

    /// Test that newly created relation is empty
    #[test]
    fn relation_initially_empty(name in "[a-z][a-z0-9]{0,5}") {
        let schema = Schema::new(name.clone());
        let rel_id = RelationId::new(1);
        let relation = Relation::new(rel_id, name, schema, RelationKind::Edb);

        prop_assert!(relation.is_empty());
        prop_assert_eq!(relation.len(), 0);
    }

    /// Test that schema with same structure is compatible with itself
    #[test]
    fn schema_self_compatible(name in "[a-z][a-z0-9]{0,5}") {
        let schema = Schema::new(name);
        prop_assert!(schema.is_compatible(&schema));
    }

    /// Test that two empty schemas are compatible
    #[test]
    fn empty_schemas_compatible(
        name1 in "[a-z][a-z0-9]{0,5}",
        name2 in "[a-z][a-z0-9]{0,5}"
    ) {
        let schema1 = Schema::new(name1);
        let schema2 = Schema::new(name2);
        prop_assert!(schema1.is_compatible(&schema2));
    }

    /// Test that relation IDs are created correctly
    #[test]
    fn relation_id_creation(id in 0u64..1000u64) {
        let rel_id = RelationId::new(id);
        prop_assert_eq!(rel_id.raw(), id);
    }

    /// Test that relation IDs with same value are equal
    #[test]
    fn relation_id_equality(id in 0u64..1000u64) {
        let rel_id1 = RelationId::new(id);
        let rel_id2 = RelationId::new(id);
        prop_assert_eq!(rel_id1, rel_id2);
    }

    /// Test that relation IDs with different values are not equal
    #[test]
    fn relation_id_inequality(id1 in 0u64..500u64, id2 in 501u64..1000u64) {
        let rel_id1 = RelationId::new(id1);
        let rel_id2 = RelationId::new(id2);
        prop_assert_ne!(rel_id1, rel_id2);
    }
}
