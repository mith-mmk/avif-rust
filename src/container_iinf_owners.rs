//! Native ownership for the `iinf` collection and its item names.

use super::container_budget::{AllocationClass, AllocationToken, ParseContext};
use super::{DecoderError, ItemInfo};

/// The original allocation tickets for item-name strings plus their backing
/// ticket vector.  Legacy parsing never constructs this sidecar.
#[derive(Debug)]
pub(super) struct NativeItemNameOwners {
    pub(super) name_tokens: Vec<AllocationToken>,
    pub(super) backing_token: AllocationToken,
}

impl NativeItemNameOwners {
    pub(super) fn new(
        context: &mut ParseContext<'_>,
        entry_count: usize,
    ) -> Result<Option<Self>, DecoderError> {
        if !context.is_native() {
            return Ok(None);
        }
        let mut name_tokens = Vec::new();
        let mut backing_token = AllocationToken::new(AllocationClass::Metadata);
        context.try_reserve_with_token(
            &mut name_tokens,
            &mut backing_token,
            entry_count,
            "iinf item-name owner tickets",
        )?;
        Ok(Some(Self {
            name_tokens,
            backing_token,
        }))
    }

    pub(super) fn push(
        &mut self,
        context: &mut ParseContext<'_>,
        token: AllocationToken,
    ) -> Result<(), DecoderError> {
        context.try_reserve_with_token(
            &mut self.name_tokens,
            &mut self.backing_token,
            1,
            "iinf item-name owner tickets",
        )?;
        self.name_tokens.push(token);
        Ok(())
    }

    fn checked_bytes(&self) -> Result<usize, DecoderError> {
        self.name_tokens.iter().try_fold(
            self.backing_token
                .checked_metadata_bytes("iinf item-name backing")?,
            |total, token| {
                total
                    .checked_add(token.checked_metadata_bytes("iinf item-name")?)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam(
                            "iinf item-name ownership size overflows".to_string(),
                        )
                    })
            },
        )
    }

    fn validate_for_retire(&self, context: &ParseContext<'_>) -> Result<usize, DecoderError> {
        let bytes = self.checked_bytes()?;
        let backing_bytes = crate::allocation::capacity_bytes::<AllocationToken>(
            self.name_tokens.capacity(),
            "iinf item-name backing",
        )?;
        if backing_bytes != self.backing_token.charged_capacity_bytes {
            return Err(DecoderError::InvalidParam(
                "iinf item-name backing token is stale".to_string(),
            ));
        }
        context.validate_release(AllocationClass::Metadata, bytes, "iinf item names")?;
        Ok(bytes)
    }

    fn release_name_tokens(
        self,
        context: &mut ParseContext<'_>,
    ) -> Result<ReleasedItemNameOwners, DecoderError> {
        let Self {
            mut name_tokens,
            backing_token,
        } = self;
        for token in &mut name_tokens {
            context.release_token(token)?;
        }
        Ok(ReleasedItemNameOwners {
            name_tokens,
            backing_token,
        })
    }
}

struct ReleasedItemNameOwners {
    name_tokens: Vec<AllocationToken>,
    backing_token: AllocationToken,
}

impl ReleasedItemNameOwners {
    fn retire_after_outer(self, context: &mut ParseContext<'_>) -> Result<(), DecoderError> {
        let Self {
            name_tokens,
            mut backing_token,
        } = self;
        drop(name_tokens);
        context.release_token(&mut backing_token)
    }
}

#[derive(Debug)]
pub(super) struct OwnedItemInfos {
    pub(super) infos: Vec<ItemInfo>,
    pub(super) outer_token: AllocationToken,
    pub(super) name_owners: Option<NativeItemNameOwners>,
}

impl OwnedItemInfos {
    pub(super) fn validate_for_retire(
        infos: &Vec<ItemInfo>,
        outer_token: &AllocationToken,
        name_owners: Option<&NativeItemNameOwners>,
        context: &ParseContext<'_>,
    ) -> Result<usize, DecoderError> {
        // Legacy parsing deliberately has no native ownership sidecar.  Its
        // historical Vec allocations are untracked, so native token/capacity
        // invariants must not reject an otherwise valid replacement.
        if !context.is_native() {
            return Ok(0);
        }
        let outer_bytes = outer_token.checked_metadata_bytes("iinf entries")?;
        let actual_outer_bytes =
            crate::allocation::capacity_bytes::<ItemInfo>(infos.capacity(), "iinf entries")?;
        if outer_bytes != actual_outer_bytes {
            return Err(DecoderError::InvalidParam(
                "iinf entries owner token is stale".to_string(),
            ));
        }
        let mut bytes = outer_bytes;
        if let Some(name_owners) = name_owners {
            if name_owners.name_tokens.len() != infos.len()
                || infos
                    .iter()
                    .zip(&name_owners.name_tokens)
                    .any(|(info, token)| {
                        crate::allocation::capacity_bytes::<u8>(
                            info.item_name.capacity(),
                            "iinf item-name",
                        ) != Ok(token.charged_capacity_bytes)
                    })
            {
                return Err(DecoderError::InvalidParam(
                    "iinf item-name owner token is stale".to_string(),
                ));
            }
            bytes = bytes
                .checked_add(name_owners.validate_for_retire(context)?)
                .ok_or_else(|| {
                    DecoderError::InvalidParam("iinf ownership size overflows".to_string())
                })?;
        }
        context.validate_release(AllocationClass::Metadata, bytes, "iinf ownership")?;
        Ok(bytes)
    }

    pub(super) fn retire(self, context: &mut ParseContext<'_>) -> Result<(), DecoderError> {
        let Self {
            infos,
            mut outer_token,
            name_owners,
        } = self;
        Self::validate_for_retire(&infos, &outer_token, name_owners.as_ref(), context)?;
        let name_bytes = name_owners
            .as_ref()
            .map(|owners| owners.validate_for_retire(context))
            .transpose()?;
        let outer_bytes = outer_token.checked_metadata_bytes("iinf entries")?;
        let total = outer_bytes
            .checked_add(name_bytes.unwrap_or(0))
            .ok_or_else(|| {
                DecoderError::InvalidParam("iinf ownership size overflows".to_string())
            })?;
        context.validate_release(AllocationClass::Metadata, total, "iinf ownership")?;
        drop(infos);
        let released_name_owners = if let Some(name_owners) = name_owners {
            Some(name_owners.release_name_tokens(context)?)
        } else {
            None
        };
        context.release_token(&mut outer_token)?;
        if let Some(name_owners) = released_name_owners {
            name_owners.retire_after_outer(context)?;
        }
        Ok(())
    }
}

trait MetadataTokenBytes {
    fn checked_metadata_bytes(&self, label: &str) -> Result<usize, DecoderError>;
}

impl MetadataTokenBytes for AllocationToken {
    fn checked_metadata_bytes(&self, label: &str) -> Result<usize, DecoderError> {
        if self.class != AllocationClass::Metadata && self.charged_capacity_bytes != 0 {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF {label} has an unexpected allocation class"
            )));
        }
        Ok(self.charged_capacity_bytes)
    }
}
