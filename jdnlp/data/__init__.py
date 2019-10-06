from jdnlp.data.transform import apply_transforms, transform_and_split

from jdnlp.data.augment import augment_transform
from jdnlp.data.entity_masking import entity_mask_transform

TRANSFORMATION_REGISTER = {
    'augment': augment_transform,
    'entity_mask': entity_mask_transform
}