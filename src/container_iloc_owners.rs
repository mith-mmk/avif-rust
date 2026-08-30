//! Native ownership for `iloc` collections and their nested extent vectors.

use super::container_budget::{AllocationClass, AllocationToken, ParseContext};
use super::{DecoderError, ItemExtent, ItemLocation};

#[derive(Debug)]
pub(super) struct ExtentOwnerTickets {
    pub(super) extents: AllocationToken,
    pub(super) indexes: AllocationToken,
}

#[derive(Debug)]
pub(super) struct NativeIlocOwners {
    pub(super) entries: Vec<ExtentOwnerTickets>,
    pub(super) backing_token: AllocationToken,
}

impl NativeIlocOwners {
    pub(super) fn new(
        context: &mut ParseContext<'_>,
        item_count: usize,
    ) -> Result<Option<Self>, DecoderError> {
        if !context.is_native() {
            return Ok(None);
        }
        let mut entries = Vec::new();
        let mut backing_token = AllocationToken::new(AllocationClass::Metadata);
        context.try_reserve_with_token(
            &mut entries,
            &mut backing_token,
            item_count,
            "iloc extent owner tickets",
        )?;
        Ok(Some(Self {
            entries,
            backing_token,
        }))
    }

    pub(super) fn push(
        &mut self,
        extents: AllocationToken,
        indexes: AllocationToken,
    ) -> Result<(), DecoderError> {
        self.entries.push(ExtentOwnerTickets { extents, indexes });
        Ok(())
    }

    fn checked_bytes(&self) -> Result<usize, DecoderError> {
        let backing = crate::allocation::capacity_bytes::<ExtentOwnerTickets>(
            self.entries.capacity(),
            "iloc extent owner tickets",
        )?;
        self.entries.iter().try_fold(backing, |total, ticket| {
            total
                .checked_add(ticket.extents.charged_capacity_bytes)
                .and_then(|total| total.checked_add(ticket.indexes.charged_capacity_bytes))
                .ok_or_else(|| DecoderError::InvalidParam("iloc owner bytes overflow".to_string()))
        })
    }

    fn validate_for_retire(
        &self,
        locations: &[ItemLocation],
        extent_indexes: &[(u32, Vec<u64>)],
        context: &ParseContext<'_>,
    ) -> Result<usize, DecoderError> {
        if self.entries.len() != locations.len() || self.entries.len() != extent_indexes.len() {
            return Err(DecoderError::InvalidParam(
                "iloc extent owner count is stale".to_string(),
            ));
        }
        let backing_bytes = crate::allocation::capacity_bytes::<ExtentOwnerTickets>(
            self.entries.capacity(),
            "iloc extent owner tickets",
        )?;
        if (self.backing_token.class != AllocationClass::Metadata
            && self.backing_token.charged_capacity_bytes != 0)
            || backing_bytes != self.backing_token.charged_capacity_bytes
        {
            return Err(DecoderError::InvalidParam(
                "iloc extent owner backing token is stale".to_string(),
            ));
        }
        for ((location, (item_id, indexes)), ticket) in
            locations.iter().zip(extent_indexes).zip(&self.entries)
        {
            if location.item_id != *item_id {
                return Err(DecoderError::InvalidParam(
                    "iloc extent owner item order is stale".to_string(),
                ));
            }
            let extent_bytes = crate::allocation::capacity_bytes::<ItemExtent>(
                location.extents.capacity(),
                "iloc extents",
            )?;
            let index_bytes = crate::allocation::capacity_bytes::<u64>(
                indexes.capacity(),
                "iloc extent indexes",
            )?;
            if (ticket.extents.class != AllocationClass::Metadata
                && ticket.extents.charged_capacity_bytes != 0)
                || (ticket.indexes.class != AllocationClass::Metadata
                    && ticket.indexes.charged_capacity_bytes != 0)
                || extent_bytes != ticket.extents.charged_capacity_bytes
                || index_bytes != ticket.indexes.charged_capacity_bytes
            {
                return Err(DecoderError::InvalidParam(
                    "iloc nested owner token is stale".to_string(),
                ));
            }
        }
        let bytes = self.checked_bytes()?;
        context.validate_release(AllocationClass::Metadata, bytes, "iloc extent owners")?;
        Ok(bytes)
    }

    fn retire_after_values_drop(self, context: &mut ParseContext<'_>) -> Result<(), DecoderError> {
        let Self {
            mut entries,
            mut backing_token,
        } = self;
        for ticket in &mut entries {
            context.release_token(&mut ticket.extents)?;
            context.release_token(&mut ticket.indexes)?;
        }
        drop(entries);
        context.release_token(&mut backing_token)
    }
}

#[derive(Debug)]
pub(super) struct OwnedItemLocations {
    pub(super) locations: Vec<ItemLocation>,
    pub(super) construction_methods: Vec<(u32, u16)>,
    pub(super) extent_indexes: Vec<(u32, Vec<u64>)>,
    pub(super) locations_token: AllocationToken,
    pub(super) construction_methods_token: AllocationToken,
    pub(super) extent_indexes_token: AllocationToken,
    pub(super) native_owners: Option<NativeIlocOwners>,
}

impl OwnedItemLocations {
    pub(super) fn validate_for_retire(
        &self,
        context: &ParseContext<'_>,
    ) -> Result<usize, DecoderError> {
        if !context.is_native() {
            return Ok(0);
        }
        if self.locations.is_empty()
            && self.construction_methods.is_empty()
            && self.extent_indexes.is_empty()
            && self.locations_token.charged_capacity_bytes == 0
            && self.construction_methods_token.charged_capacity_bytes == 0
            && self.extent_indexes_token.charged_capacity_bytes == 0
            && self.native_owners.is_none()
        {
            return Ok(0);
        }
        let outer = [
            (
                self.locations_token.class,
                self.locations_token.charged_capacity_bytes,
                crate::allocation::capacity_bytes::<ItemLocation>(
                    self.locations.capacity(),
                    "iloc locations",
                )?,
            ),
            (
                self.construction_methods_token.class,
                self.construction_methods_token.charged_capacity_bytes,
                crate::allocation::capacity_bytes::<(u32, u16)>(
                    self.construction_methods.capacity(),
                    "iloc methods",
                )?,
            ),
            (
                self.extent_indexes_token.class,
                self.extent_indexes_token.charged_capacity_bytes,
                crate::allocation::capacity_bytes::<(u32, Vec<u64>)>(
                    self.extent_indexes.capacity(),
                    "iloc extent indexes",
                )?,
            ),
        ];
        if outer.iter().any(|(class, charged, actual)| {
            *class != AllocationClass::Metadata || *charged != *actual
        }) {
            return Err(DecoderError::InvalidParam(
                "iloc outer owner token is stale".to_string(),
            ));
        }
        if self.construction_methods.len() != self.locations.len()
            || self
                .locations
                .iter()
                .zip(&self.construction_methods)
                .any(|(location, (item_id, _))| location.item_id != *item_id)
        {
            return Err(DecoderError::InvalidParam(
                "iloc method owner order is stale".to_string(),
            ));
        }
        let nested = self
            .native_owners
            .as_ref()
            .ok_or_else(|| {
                DecoderError::InvalidParam("iloc nested owners are missing".to_string())
            })?
            .validate_for_retire(&self.locations, &self.extent_indexes, context)?;
        let outer_bytes = outer.iter().try_fold(0usize, |total, (_, _, actual)| {
            total
                .checked_add(*actual)
                .ok_or_else(|| DecoderError::InvalidParam("iloc owner bytes overflow".to_string()))
        })?;
        let total = outer_bytes
            .checked_add(nested)
            .ok_or_else(|| DecoderError::InvalidParam("iloc owner bytes overflow".to_string()))?;
        context.validate_release(AllocationClass::Metadata, total, "iloc ownership")?;
        Ok(total)
    }

    pub(super) fn retire(self, context: &mut ParseContext<'_>) -> Result<(), DecoderError> {
        if !context.is_native() {
            drop(self);
            return Ok(());
        }
        if self.locations.is_empty()
            && self.construction_methods.is_empty()
            && self.extent_indexes.is_empty()
            && self.native_owners.is_none()
        {
            drop(self);
            return Ok(());
        }
        self.validate_for_retire(context)?;
        let Self {
            locations,
            construction_methods,
            extent_indexes,
            mut locations_token,
            mut construction_methods_token,
            mut extent_indexes_token,
            native_owners,
        } = self;
        drop(locations);
        drop(construction_methods);
        drop(extent_indexes);
        let native_owners = native_owners.expect("validated Native iloc owners");
        native_owners.retire_after_values_drop(context)?;
        context.release_token(&mut locations_token)?;
        context.release_token(&mut construction_methods_token)?;
        context.release_token(&mut extent_indexes_token)
    }
}
