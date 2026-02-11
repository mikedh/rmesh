// PBR mesh shader - Cook-Torrance BRDF with GGX distribution
// Full PBR pipeline: base color, metallic-roughness, normal map, occlusion, emissive

#include "uniforms.inc.wgsl"

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;
@group(2) @binding(0) var<uniform> material: MaterialUniforms;

@group(3) @binding(0) var base_color_tex: texture_2d<f32>;
@group(3) @binding(1) var mr_tex: texture_2d<f32>;
@group(3) @binding(2) var normal_tex: texture_2d<f32>;
@group(3) @binding(3) var occlusion_tex: texture_2d<f32>;
@group(3) @binding(4) var emissive_tex: texture_2d<f32>;
@group(3) @binding(5) var tex_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tangent: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) world_tangent: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model.model * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    out.world_normal = normalize((model.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);
    let wt = normalize((model.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
    out.world_tangent = vec4<f32>(wt, in.tangent.w);
    out.color = in.color;
    out.uv = in.uv;
    return out;
}

// PBR helper functions
const PI: f32 = 3.14159265359;

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    return ggx1 * ggx2;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let max_reflect = vec3<f32>(1.0 - roughness);
    return f0 + (max(max_reflect, f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var albedo = material.base_color.rgb;
    var alpha = material.base_color.a;

    // Sample base color texture if present
    if material.has_texture != 0u {
        let tex_color = textureSample(base_color_tex, tex_sampler, in.uv);
        albedo *= tex_color.rgb;
        alpha *= tex_color.a;
    }

    if material.use_vertex_color != 0u {
        albedo = in.color.rgb;
        alpha = in.color.a;
    }

    // Metallic-roughness from texture or uniforms
    var metallic = material.metallic;
    var roughness = material.roughness;
    if material.has_mr_texture != 0u {
        let mr_sample = textureSample(mr_tex, tex_sampler, in.uv);
        metallic *= mr_sample.b;   // glTF spec: B=metallic
        roughness *= mr_sample.g;  // glTF spec: G=roughness
    }
    roughness = max(roughness, 0.04);

    // Normal mapping
    var n = normalize(in.world_normal);
    if material.has_normal_texture != 0u {
        let normal_sample = textureSample(normal_tex, tex_sampler, in.uv).xyz;
        let tangent_normal = (normal_sample * 2.0 - 1.0) * vec3<f32>(material.normal_scale, material.normal_scale, 1.0);
        let t = normalize(in.world_tangent.xyz);
        let b = normalize(cross(n, t) * in.world_tangent.w);
        // TBN matrix transforms tangent-space normal to world space
        n = normalize(t * tangent_normal.x + b * tangent_normal.y + n * tangent_normal.z);
    }

    let v = normalize(camera.camera_pos.xyz - in.world_pos);

    // F0 for Fresnel
    var f0 = vec3<f32>(0.04);
    f0 = mix(f0, albedo, metallic);

    // Directional light
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let light_color = vec3<f32>(1.0, 1.0, 1.0);
    let light_intensity = 3.0;

    let l = light_dir;
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);

    // Cook-Torrance BRDF
    let ndf = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let numerator = ndf * g * f;
    let denominator = 4.0 * max(dot(n, v), 0.0) * n_dot_l + 0.0001;
    let specular = numerator / denominator;

    let ks = f;
    let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallic);

    var lo = (kd * albedo / PI + specular) * light_color * light_intensity * n_dot_l;

    // Fill light from opposite side (dimmer)
    let fill_dir = normalize(vec3<f32>(-0.3, 0.5, -0.5));
    let fill_n_dot_l = max(dot(n, fill_dir), 0.0);
    lo += albedo * 0.3 * fill_n_dot_l;

    // Ambient / environment lighting
    var ambient = vec3<f32>(0.0);
    if camera.use_env_light != 0u {
        // Hemisphere sky approximation
        let sky_top = vec3<f32>(0.6, 0.7, 0.9);
        let sky_bottom = vec3<f32>(0.2, 0.2, 0.25);
        let sky_blend = n.y * 0.5 + 0.5;
        let irradiance = mix(sky_bottom, sky_top, sky_blend);

        let n_dot_v = max(dot(n, v), 0.0);
        let f_env = fresnel_schlick_roughness(n_dot_v, f0, roughness);
        let kd_env = (vec3<f32>(1.0) - f_env) * (1.0 - metallic);

        // Diffuse ambient
        ambient = kd_env * albedo * irradiance;

        // Specular ambient (reflection probe approximation)
        let reflect_dir = reflect(-v, n);
        let reflect_blend = reflect_dir.y * 0.5 + 0.5;
        let blur = roughness * roughness;
        let env_color = mix(
            mix(sky_bottom, sky_top, reflect_blend),
            irradiance,
            blur
        );
        ambient += f_env * env_color;
    } else {
        ambient = vec3<f32>(0.03) * albedo;
    }

    // Occlusion
    var ao = 1.0;
    if material.has_occlusion_texture != 0u {
        let ao_sample = textureSample(occlusion_tex, tex_sampler, in.uv).r;
        ao = mix(1.0, ao_sample, material.occlusion_strength);
    }

    // Emissive
    var emissive = vec3<f32>(0.0);
    if material.has_emissive_texture != 0u {
        emissive = textureSample(emissive_tex, tex_sampler, in.uv).rgb * material.emissive_factor;
    } else {
        emissive = material.emissive_factor;
    }

    var color = ambient * ao + lo + emissive;

    // Reinhard tonemap
    color = color / (color + vec3<f32>(1.0));
    // Note: no manual gamma — the sRGB render target handles linear→sRGB conversion

    return vec4<f32>(color, alpha);
}
