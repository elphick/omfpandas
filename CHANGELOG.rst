Omfpandas 0.9.0 (2025-03-23)
============================

Feature
-------

- Added aggregate function to utils (#52)


Other Tasks
-----------

- Moved create_test_blockmodel to pandas_utils (#52)


Omfpandas 0.8.3 (2025-03-17)
============================

Feature
-------

- Added optional encoded integer index to regular blockmodel dataframe (#50)


Omfpandas 0.8.2 (2025-03-11)
============================

Feature
-------

- Added composite element support for block models.  Uses dot notation: composite_name.blockmodel_name (#48)


Omfpandas 0.8.1 (2025-02-25)
============================

Feature
-------

- Added test_block_model utility function. (#41)


Other Tasks
-----------

- Renamed utils modules to avoid name conflicts. (#41)


Omfpandas 0.8.0 (2025-02-24)
============================

Bugfix
------

- Corrected MultiIndex order to C-order (x, y, z). (#39)


Other Tasks
-----------

- Added blockmodel attribute sorting validation example. (#39)
- Removed __omf_version__ (#39)
- Removed OMF1 support for simplicity. (#39)


Omfpandas 0.7.0 (2025-02-19)
============================

Feature
-------

- Breaking changes to support omf and omf2.  Modified tests and examples. (#34)


Omfpandas 0.6.11 (2024-11-18)
=============================

Doc
---

- Improved examples (#30)


Omfpandas 0.6.10 (2024-09-20)
=============================

Feature
-------

- Validation during writing can now be via a dict as well as a schema yaml file. (#28)


Omfpandas 0.6.9 (2024-08-20)
============================

Bugfix
------

- Fixed no-file-exists bug for writer.  Added unit test. (#26)


Omfpandas 0.6.8 (2024-08-11)
============================

Feature
-------

- Added calculated blockmodel attributes. Example pending. (#23)


Omfpandas 0.6.7 (2024-08-11)
============================

Other Tasks
-----------

- Merging old branches.


Omfpandas 0.6.6 (2024-08-11)
============================

Feature
-------

- Added query to read_blockmodels.  Prioritises memory over speed. (#21)
