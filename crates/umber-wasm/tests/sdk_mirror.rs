//! The guest SDK's wire mirror agrees with the host's `wire.rs`.
//!
//! Sharing approach: `src/wire_consts.rs` is copied VERBATIM into
//! `crates/umber-plugin-sdk/src/wire_consts.rs` (the SDK can't depend on
//! this wasmtime crate, and a cross-crate `#[path]` include breaks
//! packaging). The first test pins the copies byte-identical; the rest
//! cross-encode through the SDK compiled for the host.

use umber_graph::{ImageBuffer, ParamValue};
use umber_plugin_sdk::{
    decode_input as sdk_decode, eval_slices, node_def_record, Image, ImageMut, NodeError, Param,
    Params,
};
use umber_wasm::wire::{self, NodeDef};

#[test]
fn sdk_wire_consts_are_byte_identical() {
    let host = include_str!("../src/wire_consts.rs");
    let sdk = include_str!("../../umber-plugin-sdk/src/wire_consts.rs");
    assert!(
        host == sdk,
        "crates/umber-plugin-sdk/src/wire_consts.rs drifted from \
         crates/umber-wasm/src/wire_consts.rs — re-copy it verbatim"
    );
    // And the compiled values agree (guards a stale build too).
    assert_eq!(
        umber_wasm::wire_consts::WIRE_MAGIC,
        umber_plugin_sdk::wire_consts::WIRE_MAGIC
    );
}

fn as_sdk(v: &ParamValue) -> Param<'_> {
    match v {
        ParamValue::Float(f) => Param::Float(*f),
        ParamValue::Vec2(a) => Param::Vec2(*a),
        ParamValue::Vec3(a) => Param::Vec3(*a),
        ParamValue::Color(a) => Param::Color(*a),
        ParamValue::Int(i) => Param::Int(*i),
        ParamValue::Bool(b) => Param::Bool(*b),
        ParamValue::Asset(s) => Param::Asset(s),
        ParamValue::NodeRef(_) => unreachable!("never on the wire"),
    }
}

#[test]
fn host_encoded_input_decodes_identically_in_the_sdk() {
    let data: Vec<u8> = (0..5 * 3 * 4).map(|i| (i * 13 % 256) as u8).collect();
    let img = ImageBuffer::new(5, 3, data).unwrap();
    for params in [
        vec![],
        vec![
            ("strength".to_string(), ParamValue::Float(0.3)),
            ("center".into(), ParamValue::Vec2([0.5, 0.25])),
            ("tint".into(), ParamValue::Color([1.0, 0.0, 0.5])),
            ("flip".into(), ParamValue::Bool(true)),
        ],
        vec![
            ("up".to_string(), ParamValue::Vec3([0.0, 1.0, 0.0])),
            ("r".into(), ParamValue::Int(-7)),
            ("tex".into(), ParamValue::Asset("assets/ab/cd.png".into())),
            ("wired".into(), ParamValue::NodeRef(4)), // filtered out
        ],
    ] {
        let bytes = wire::encode_input(&img, &params).unwrap();
        let (simg, sparams) = sdk_decode(&bytes).expect("SDK accepts the host's region");
        assert_eq!((simg.width, simg.height), (img.width, img.height));
        assert_eq!(simg.data, &img.data[..]);
        let sent = wire::wire_params(&params);
        assert_eq!(sparams.len(), sent.len());
        for (i, (name, value)) in sent.iter().enumerate() {
            assert_eq!(sparams.nth(i), Some((name.as_str(), as_sdk(value))));
            assert_eq!(sparams.get(name), Some(as_sdk(value)));
        }
    }
}

#[test]
fn sdk_rejects_what_the_host_rejects() {
    let img = ImageBuffer::filled(2, 2, [1, 2, 3, 4]).unwrap();
    let good = wire::encode_input(&img, &[("k".into(), ParamValue::Int(1))]).unwrap();
    let mut bad = good.clone();
    bad[1] ^= 0xFF;
    assert!(wire::decode_input(&bad).is_err());
    assert_eq!(sdk_decode(&bad).unwrap_err(), NodeError::BadMagic);
    for cut in [3, 12, good.len() - 1] {
        assert!(wire::decode_input(&good[..cut]).is_err(), "host, cut {cut}");
        assert!(sdk_decode(&good[..cut]).is_err(), "sdk, cut {cut}");
    }
}

fn swap_rb(i: &Image<'_>, _: &Params<'_>, o: &mut ImageMut<'_>) -> Result<(), NodeError> {
    for (o, i) in o.data.chunks_exact_mut(4).zip(i.data.chunks_exact(4)) {
        o.copy_from_slice(&[i[2], i[1], i[0], i[3]]);
    }
    Ok(())
}

#[test]
fn sdk_out_region_decodes_on_the_host() {
    let data: Vec<u8> = (0..3 * 2 * 4).map(|i| i as u8).collect();
    let img = ImageBuffer::new(3, 2, data).unwrap();
    let input = wire::encode_input(&img, &[]).unwrap();
    let mut out = vec![0u8; wire::out_capacity(&img)];
    let n = eval_slices(&input, &mut out, swap_rb);
    assert_eq!(n as usize, out.len());
    let back = wire::decode_output(&out[..n as usize]).expect("host accepts the SDK's region");
    assert_eq!((back.width, back.height), (3, 2));
    assert_eq!(back.pixel(0, 0), Some([2, 1, 0, 3]));
    assert_eq!(back.pixel(2, 1), Some([22, 21, 20, 23]));
    // And the out region the SDK wrote is exactly the host's encoding.
    assert_eq!(&out[..], &wire::encode_output(&back)[..]);
}

#[test]
fn sdk_node_def_record_decodes_on_the_host() {
    const REC: [u8; 24] = node_def_record::<24>("vignette", 1);
    let def = wire::decode_node_def(&REC).expect("host accepts the SDK's record");
    let expected = NodeDef {
        name: "vignette".into(),
        n_inputs: 1,
        n_params: 1,
    };
    assert_eq!(def, expected);
    assert_eq!(&REC[..], &wire::encode_node_def(&expected).unwrap()[..]);
}
