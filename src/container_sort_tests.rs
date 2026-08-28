use super::container_budget::ParseContext;
use super::*;

fn limits(items: usize) -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        1 << 20,
        64,
        64,
        4096,
        8192,
        1 << 20,
        1 << 20,
        items,
        items,
        32,
        8,
        1,
    )
}

#[test]
fn native_unique_alpha_selection_is_sorted_without_duplicate_records() {
    let state = MetaState {
        item_references: vec![ItemReference {
            reference_type: *b"auxl",
            from_item_id: 1,
            to_item_ids: vec![4, 2, 3],
        }],
        item_infos: [2, 3, 4]
            .into_iter()
            .map(|item_id| ItemInfo {
                item_id,
                item_type: *b"av01",
                item_name: String::new(),
            })
            .collect(),
        item_locations: [2, 3, 4]
            .into_iter()
            .map(|item_id| ItemLocation {
                item_id,
                base_offset: 0,
                extents: Vec::new(),
            })
            .collect(),
        ..MetaState::default()
    };
    let bounds = limits(4);
    let mut context = ParseContext::native_still(&bounds);

    let output =
        super::alpha_auxiliary_items_for_with_context(&[], &state, Some(1), &mut context).unwrap();

    assert_eq!(
        output.iter().map(|item| item.item_id).collect::<Vec<_>>(),
        vec![2, 3, 4]
    );
    assert_eq!(
        output
            .iter()
            .map(|item| item.aux_type.as_str())
            .collect::<Vec<_>>(),
        vec![ALPHA_AUX_TYPE; 3]
    );
}
