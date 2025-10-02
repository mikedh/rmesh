A collection of schemas for mesh formats. General thought is to use something like [xsd_parser](https://docs.rs/xsd-parser/latest/xsd_parser/) with `quick-xml` for validating files before we start looking at the data.


GLTF:

```
import trimesh, zstandard
schema_json = json.dumps(trimesh.exchange.gltf.get_schema(), separators=(',', ':')
schema_comp = zstandard.compress(schema_json, 23)
```