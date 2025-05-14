Shader "Unlit/GenShin_Render"
{
    Properties //着色器的输入 
    {
        _BaseMap ("Base Color", 2D) = "white" {}
        _AlphaTex("Alpha Tex", 2D) = "white"
        _ILMTex ("ILM Texture", 2D) = "white" {}
        _DiffuseColor ("Diffuse Color", Color) = (1,1,1,1)
        _AmbientColor ("Ambient Color", Color) = (0.5,0.5,0.5,1)
        _ShadowColor ("Shadow Color", Color) = (0.7,0.7,0.7,1)
        _AmbientIntensity ("Ambient Intensity", Range(0, 1)) = 0.5
        _NormalMap ("Normal Map", 2D) = "bump" {}
        _BumpScale ("Normal Scale", Range(0, 1)) = 1.0
        _RampTex ("Ramp Texture", 2D) = "white" {}

        _ToonTex("Toon Tex", 2D) = "white" {}

        _BaseTexFac("BaseTexFac", Range(0, 1)) = 1
        _ToonTexFac("ToonTexFac", Range(0, 1)) = 1
        _SphereTexFac("SphereTexFac", Range(0, 1)) = 0
        _SphereMulAdd("SphereMul/Add", Range(0, 1)) = 0

        _SpecExpon("Specular Expon", Range(0, 100)) = 50
        _KsNonMetallic("KsNonMetallic", Range(0, 3)) = 1
        _KsMetallic("KsMetallic", Range(0, 3)) = 1
        _MetallicTex("MetallicTex", 2D) = "white" {}

        _RampMapRow0("RampMapRow0", Range(1,5)) = 1
        _RampMapRow1("RampMapRow1", Range(1,5)) = 4
        _RampMapRow2("RampMapRow2", Range(1,5)) = 3
        _RampMapRow3("RampMapRow3", Range(1,5)) = 5
        _RampMapRow4("RampMapRow4", Range(1,5)) = 2

        [Header(ScreenRim)]
        _RimOffset("Rim Offset", Range(0, 10)) = 4
        _RimThreshold("Rim Threshold", Range(0, 1)) = 0.03
        _RimColor("Rim Color", Color) = (1, 1, 1, 1)
        _RimIntensity("Rim Intensity", Range(0, 10)) = 1

        [Header(OutLine)]
        _OutLineWidth("OutLine Width", Range(0,5)) = 0.1
        _MaxOutlineZoffset("Max Outline Zoffset", Range(0,0.1)) = 0.01
        _OutlineColor0("OutLine Color0", Color) = (0,0,0,1)
        _OutlineColor1("OutLine Color1", Color) = (0,0,0,1)
        _OutlineColor2("OutLine Color2", Color) = (0,0,0,1)
        _OutlineColor3("OutLine Color3", Color) = (0,0,0,1)
        _OutlineColor4("OutLine Color4", Color) = (0,0,0,1)


        [Header(Screen Space Shadow)]
        [Toggle(_SCREEN_SPACE_SHADOW)] _ScreenSpaceShadow("Screen Space Shadow", Float) = 1
        _ScreenSpaceShadowWidth("Screen Space Shadow Width", Range(0, 1)) = 0.2
        _ScreenSpaceShadowFadeout("Screen Space Shadow Fadeout", Range(0, 1)) = 0.015
        _ScreenSpaceShadowThreshold("Screen Space Shadow Threshold", Range(0, 10)) = 0.2

        [Header(Flow Light)]
        [Toggle(_FLOWLight)] _FlowLight("Flow Light", Float) = 1
        _FlowMask("Flow Mask", 2D) = "white" {}
        _FlowTillingSpeed("Flow Tilling Speed", Float) = (1, 1, 0, 0)
        _FlowMap("Flow Map", 2D) = "white" {}
        _FlowRamp("Flow Ramp", 2D) = "white" {}

        [Header(Nyx fire)]
        [Toggle(_NyxFire)] _NyxFire("Nyx Fire",Float) = 1
        _NyxNoise("Nyx Noise", 2D) = "white" {}
        _NyxRamp("Nyx Ramp", 2D) = "white" {}
        _NyxOutLineWidth("OutLine Width", Range(0,10)) = 0.1
        _NyxMaxOutlineZoffset("Max Outline Zoffset", Range(0,10)) = 0.01
        _NyxNoiseSpeed("Nyx Noise Speed", Float) = (1, 1, 0, 0)
        _NyxNoiseIntensity("Nyx Noise Intensity", Range(0, 10)) = 1
        _NyxLineSpeed("Nyx Line Speed", Float) = (1,1,0,0)


        _SphereTex("Sphere Tex", 2D) = "white" {}
        
    }
    SubShader
    {
        LOD 100

        Pass
        {
        Name "ForwardBase" //渲染通道名称  

        Tags {
            "LightMode"="UniversalForward"
            "RenderType"="Opaque"
            "RenderPipeLine"="UniversalRenderPipeline" //用于指明使用URP来渲染
        }
        Stencil
    {
        Ref 2
        Comp Always
        Pass Replace
    }
        
        ZWrite On
        Cull Off
        Blend SrcAlpha OneMinusSrcAlpha

        HLSLPROGRAM

        #pragma vertex vert
        #pragma fragment frag
        #pragma shader_feature_local _SCREEN_SPACE_SHADOW
        #pragma shader_feature_local _FLOWLight
        
        #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
        #pragma multi_compile _ _MAIN_LIGHT_SHADOWS_CASCADE
        #pragma multi_compile _ _SHADOWS_SOFT

        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl" 
        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/UnityInstancing.hlsl" 
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareNormalsTexture.hlsl"

        CBUFFER_START(UnityPerMaterial) //声明变量
            float4 _BaseMap_ST;
            float4 _NormalMap_ST;
            float _BumpScale;
            float4 DiffuseColor;
            float _BaseTexFac;
            float _ToonTexFac;
            float _SphereTexFac;
            float _SphereMulAdd;
            float4 _AmbientColor;
            float _AmbientIntensity;
            float _RampMapRow0;
            float _RampMapRow1;
            float _RampMapRow2;
            float _RampMapRow3;
            float _RampMapRow4;
            float4 _ShadowColor;
            float _ScreenSpaceShadowWidth;
            float _ScreenSpaceShadowFadeout;
            float _ScreenSpaceShadowThreshold;
            float _SpecExpon;
            float _KsNonMetallic;
            float _KsMetallic;
            float4 _FlowTillingSpeed;
            float _RimOffset;
            float4 _RimColor;
            float _RimIntensity;
            float _RimThreshold;
        CBUFFER_END

        TEXTURE2D(_BaseMap); //贴图采样  
        SAMPLER(sampler_BaseMap);
        TEXTURE2D(_NormalMap);
        SAMPLER(sampler_NormalMap);
        TEXTURE2D(_ILMTex);
        SAMPLER(sampler_ILMTex);
        TEXTURE2D(_ToonTex);
        SAMPLER(sampler_ToonTex);
        TEXTURE2D(_SphereTex);
        SAMPLER(sampler_SphereTex);
        TEXTURE2D(_RampTex);
        SAMPLER(sampler_RampTex);
        TEXTURE2D(_MetallicTex);
        SAMPLER(sampler_MetallicTex);
        TEXTURE2D(_AlphaTex);
        SAMPLER(sampler_AlphaTex);
        TEXTURE2D(_FlowMask);
        SAMPLER(sampler_FlowMask);
        TEXTURE2D(_FlowMap);
        SAMPLER(sampler_FlowMap);
        TEXTURE2D(_FlowRamp);
        SAMPLER(sampler_FlowRamp);

        float max3(float a, float b, float c)
{
    return max(max(a, b), c);
}

        float AverageColor(float3 color)
            {
                return dot(color,float3(1.0,1.0,1.0))/3.0;
            }

            float3 NormalizeColorByAverage(float3 color)
            {
                float average = AverageColor(color);
                return color / max(average,1e-5);
            }

            float3 ScaleColorByMax(float3 color)
            {
                float maxComponent = max3(color.r,color.g,color.b);
                maxComponent = min(maxComponent,1.0);
                return float3(color * maxComponent);
            }

       struct Attributes 
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float4 tangentOS : TANGENT;
                float2 uv : TEXCOORD0;
                float2 uv2 : TEXCOORD1;
                float4 vertexColor : COLOR;
            };

        struct Varyings
            {
                float2 uv : TEXCOORD0;
                float3 positionWS : TEXCOORD1;
                float3 positionVS : TEXCOORD2;
                float4 positionCS : SV_POSITION;
                float4 positionNDC : TEXCOORD3;
                float3 normalWS : TEXCOORD4;
                float3 tangentWS : TEXCOORD5;
                float3 bitangentWS : TEXCOORD6;
                float2 uv2 : TEXCOORD7;
                half4 vertexColor: COLOR;
            };

            Varyings vert (Attributes input)//顶点着色器
            {
                Varyings output;

                VertexPositionInputs vertexInput = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionNDC = vertexInput.positionNDC;

                // 初始化 positionCS
                output.positionCS = TransformObjectToHClip(input.positionOS);

                // 初始化 uv
                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                output.uv2 = TRANSFORM_TEX(input.uv2, _BaseMap);
                

                // 初始化 vertexColor
                output.vertexColor = input.vertexColor;

                // 初始化其他字段
                output.positionWS = TransformObjectToWorld(input.positionOS).xyz;
                output.positionVS = TransformWorldToView(output.positionWS);
                
                output.normalWS = TransformObjectToWorldNormal(input.normalOS);
                output.tangentWS = TransformObjectToWorldDir(input.tangentOS.xyz);
                output.bitangentWS = cross(output.normalWS, output.tangentWS) * input.tangentOS.w;

                return output;
            }

            half4 frag (Varyings input) : SV_Target  // 片元着色器
            {

                //NormalMap 法线纹理采样
                float4 PackedNormal = SAMPLE_TEXTURE2D(_NormalMap, sampler_NormalMap, input.uv);
                float3 pixelNormalTS = UnpackNormalScale(PackedNormal, _BumpScale);
                float3x3 TBN = float3x3(input.tangentWS, input.bitangentWS, input.normalWS);
                float3 pixelNormal = normalize(mul(pixelNormalTS, TBN));

                //向量计算
                float3 N = normalize(input.normalWS);
                N = pixelNormal;

                float4 shadowCoord = TransformWorldToShadowCoord(input.positionWS);
                Light mainLight = GetMainLight(shadowCoord);
                float3 lightColor = mainLight.color.rgb;
                lightColor = ScaleColorByMax(lightColor);

                float3 V = GetWorldSpaceNormalizeViewDir(input.positionWS);
                float3 L = normalize(mainLight.direction);
                half3 H = normalize(L + V);


                float NoH = dot(N, H);
                float NoV = saturate(dot(N, V));
                float NoL = saturate(dot(N, L));


                
                //Screen Space Shadow
                 float shadowAttenuation = 1.0;
                
                #if _SCREEN_SPACE_SHADOW
                {
                    // 计算线性眼空间深度
                    float linearEyeDepth = input.positionCS.w;
                    float perspective = 1.0 / linearEyeDepth;

                    // 计算阴影偏移量
                    float offsetMul = _ScreenSpaceShadowWidth * 5.0 * perspective / 100.0;

                    float3 lightDirectionVS = TransformWorldToViewDir(normalize(L));
                    float2 offset =  lightDirectionVS.xy * offsetMul;// 计算偏移

                    // 计算偏移后的屏幕坐标
                    int2 coord = input.positionCS.xy + offset * _ScaledScreenParams.xy;
                    coord = min(max(0,coord), _ScaledScreenParams.xy - 1);
                    float offsetSceneDepth = LoadSceneDepth(coord);
                    float offsetSceneLinearEyeDepth = LinearEyeDepth(offsetSceneDepth, _ZBufferParams);

                    float fadeout = max(1e-5, _ScreenSpaceShadowFadeout);
                    shadowAttenuation = saturate((offsetSceneLinearEyeDepth - (linearEyeDepth - _ScreenSpaceShadowThreshold)) * 50 / fadeout);
                }
                #endif

                float3 normalVS = TransformWorldToViewDir(N);
                float2 matCapUV = normalVS.xy * 0.5 + 0.5;
                
                //Color Mapping

                float4 BaseTex = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv);
                float4 ToonTex = SAMPLE_TEXTURE2D(_ToonTex,sampler_ToonTex,matCapUV);
                float4 SphereTex = SAMPLE_TEXTURE2D(_SphereTex, sampler_SphereTex, matCapUV);

                float3 baseColor = _AmbientColor.rgb;
                baseColor = saturate(lerp(baseColor, baseColor + DiffuseColor.rgb, 0.6));
                baseColor = lerp(baseColor, baseColor * BaseTex.rgb, _BaseTexFac);
                baseColor = lerp(baseColor, baseColor * ToonTex.rgb, _ToonTexFac);
                baseColor = lerp(lerp(baseColor, baseColor * SphereTex.rgb, _SphereTexFac), lerp(baseColor, baseColor + SphereTex.rgb, _SphereTexFac), _SphereMulAdd);


                //ILM
                float4 ILMTex = SAMPLE_TEXTURE2D(_ILMTex, sampler_ILMTex, input.uv);
                //RampV
                float MatEnum0 = 0.0;
                float MatEnum1 = 0.3;
                float MatEnum2 = 0.5;
                float MatEnum3 = 0.7;
                float MatEnum4 = 1.0;

                float Ramp0 = _RampMapRow0/10 - 0.05;
                float Ramp1 = _RampMapRow1/10 - 0.05;
                float Ramp2 = _RampMapRow2/10 - 0.05;
                float Ramp3 = _RampMapRow3/10 - 0.05;
                float Ramp4 = _RampMapRow4/10 - 0.05; 

                float dayRampV = lerp(Ramp4, Ramp3, step(ILMTex.a, (MatEnum3 + MatEnum4) / 2));
                dayRampV = lerp(dayRampV, Ramp2, step(ILMTex.a, (MatEnum2 + MatEnum3) / 2));
                dayRampV = lerp(dayRampV, Ramp1, step(ILMTex.a, (MatEnum1 + MatEnum2) / 2));
                dayRampV = lerp(dayRampV, Ramp0, step(ILMTex.a, (MatEnum0 + MatEnum1) / 2));
                float nightRampV = dayRampV + 0.5;



                //直接光
                float Lambert = max(0, NoL * shadowAttenuation);
                float HalfLambert = pow(Lambert * 0.5 + 0.5, 2);
                float LambertStep = smoothstep(0.423, 0.450, HalfLambert);

                float rampGrayU = clamp(smoothstep(0.2, 0.4, HalfLambert),0.003,0.997);
                float2 rampGrayDayUV = float2(rampGrayU, 1 - dayRampV);
                float2 rampGrayNightUV = float2(rampGrayU, 1 - nightRampV);

                float rampDrakU = 0.003;
                float2 rampDarkDayUV = float2(rampDrakU, 1 - dayRampV);
                float2 rampDarkNightUV = float2(rampDrakU, 1 - nightRampV);

                float Day = (L.y + 1)/2;
                float3 rampGrayColor = lerp(SAMPLE_TEXTURE2D(_RampTex, sampler_RampTex, rampGrayNightUV).rgb, SAMPLE_TEXTURE2D(_RampTex, sampler_RampTex, rampGrayDayUV).rgb, Day);
                float3 rampDarkColor = lerp(SAMPLE_TEXTURE2D(_RampTex, sampler_RampTex, rampDarkNightUV).rgb, SAMPLE_TEXTURE2D(_RampTex, sampler_RampTex, rampDarkDayUV).rgb, Day);

                float3 GrayShadowColor = baseColor * rampGrayColor * _ShadowColor.rgb;
                float3 DarkShadowColor = baseColor * rampDarkColor * _ShadowColor.rgb;

                float3 diffuse = 0;
                diffuse = lerp(GrayShadowColor, baseColor, LambertStep);
                diffuse = lerp(DarkShadowColor, diffuse, saturate(ILMTex.g * 2));
                diffuse = lerp(diffuse, baseColor, saturate(ILMTex.g - 0.5) * 2);
                //diffuse *= lightColor;

                //Highlight
                float blinnPhong = step(0,NoL) * pow(max(0,NoH),_SpecExpon);
                float3 NonMetallicSpec = step(1.04 - blinnPhong, ILMTex.b) * ILMTex.r * _KsNonMetallic;
                float3 MetallicSpec = blinnPhong * ILMTex.b * (LambertStep * 0.8 + 0.2) * baseColor * _KsMetallic;

                float isMetallic = step(0.95, ILMTex.r);

                float3 Specular = lerp(NonMetallicSpec, MetallicSpec, isMetallic);

                float3 Metallic = lerp(0, SAMPLE_TEXTURE2D(_MetallicTex, sampler_MetallicTex, matCapUV).r * baseColor, isMetallic);

                

                
                //float specular = SAMPLE_TEXTURE2D(_ILMTex, sampler_ILMTex, input.uv).r;

                
                //Flow Light
                float FlowMaskBody = 0;
                float FlowMaskHair = 0;
                float FlowMask = 0;
                float FlowMap = 0;
                float3 FlowColor = 0;
                #if _FLOWLight
                {
                    FlowMaskBody = SAMPLE_TEXTURE2D(_FlowMask, sampler_FlowMask, input.uv).r;
                    FlowMaskHair = SAMPLE_TEXTURE2D(_FlowMask, sampler_FlowMask, input.uv).g;
                    FlowMask = SAMPLE_TEXTURE2D(_FlowMask, sampler_FlowMask, input.uv).r + SAMPLE_TEXTURE2D(_FlowMask, sampler_FlowMask, input.uv).g;
                    float2 a = ((input.positionWS).xyz - TransformObjectToWorld(float3(0, 0, 0)).xyz).xy + (NoV * 0.1 + 0.5);
                    a *= _FlowTillingSpeed.xy; 
                    float2 pannerUV = a + _Time.y * _FlowTillingSpeed.zw; 
                    FlowMap = SAMPLE_TEXTURE2D(_FlowMap, sampler_FlowMap, pannerUV).g;
                    FlowMap = saturate(FlowMap); // 保证在0~1
                    float2 FlowRampUV = float2(FlowMap, 0.5);
                    FlowColor = SAMPLE_TEXTURE2D(_FlowRamp, sampler_FlowRamp, FlowRampUV).rgb;
                    FlowColor *= FlowMask;
                    FlowColor = ScaleColorByMax(FlowColor);
                }
                #endif

                

                // // 计算 FlowColor 的强度（可用 FlowMask 或 FlowColor 的亮度）
                // float flowStrength = saturate(length(FlowColor)); // 或直接用 FlowMask

                // // 用 flowStrength 控制插值
                // float3 finalAlbedo = lerp(albedo, FlowColor, flowStrength);

                //Screen Rim
                float2 screenUV = input.positionNDC.xy / input.positionNDC.w;
                float rawDepth = SampleSceneDepth(screenUV);
                float linearDepth = LinearEyeDepth(rawDepth, _ZBufferParams);
                float2 screenOffset = float2(lerp(-1, 1, step(0, normalVS.x)) * _RimOffset / _ScreenParams.x / max(1,pow(linearDepth, 2)), 0);
                float offsetDepth = SampleSceneDepth(screenUV + screenOffset);
                float offsetLinearDepth = LinearEyeDepth(offsetDepth, _ZBufferParams);

                float Rim = saturate(offsetLinearDepth - linearDepth);
                Rim = step(_RimThreshold, Rim) * _RimIntensity;
                Rim *= _RimColor * baseColor;
                
                float fresnelPower = 6;
                float fresnelClamp = 0.8;
                float fresnel = 1 - saturate(NoV);
                fresnel = pow(fresnel, fresnelPower);
                fresnel = fresnel * fresnelClamp + (1 - fresnelClamp);


                float3 albedo = diffuse + Specular + Metallic;
                albedo =  1 - (1 - Rim * fresnel) * (1 - albedo);
                albedo +=  FlowColor;

                
                //环境光
                float3 ambient = SampleSH(N);
                ambient = lerp(ambient, baseColor, _AmbientIntensity);

                float Alpha = SAMPLE_TEXTURE2D(_AlphaTex,sampler_AlphaTex,input.uv).a;


                half4 col = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv);
                //half res=lerp(input.vertexColor,col, input.vertexColor.g);
                return float4(albedo,Alpha);
            }

        ENDHLSL
        }
        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }
            ZWrite [_ZWrite]
            ColorMask 0
            Cull [_Cull]

            HLSLPROGRAM

            #pragma mulit_compile_instancing
            #pragma mulit_compile_DOTS_INSTANCING_ON

            #pragma vertex vert
            #pragma fragment fragDepth

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
            };

            float _AlpahClip;

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                return OUT;
            }

            float4 fragDepth(Varyings IN) : SV_Target
            {
                clip(1.0 - _AlpahClip);
                return 0;
            }
            ENDHLSL
        }
        Pass
        {
            Name "DepthNormals"
            Tags { "LightMode" = "DepthNormals" }
            ZWrite [_ZWrite]
            Cull [_Cull]

            HLSLPROGRAM

            #pragma mulit_compile_instancing
            #pragma mulit_compile_DOTS_INSTANCING_ON

            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float4 tangentOS : TANGENT;
                float3 normalOS : NORMAL;
                float2 texcoord : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float4 tangentWS : TEXCOORD2;
            };

            float _AlpahClip;

            Varyings vert(Attributes input)
            {
                Varyings output = (Varyings)0;

                output.uv = input.texcoord;
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);

                VertexPositionInputs vertexInput = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normalInput = GetVertexNormalInputs(input.normalOS, input.tangentOS);

                float3 viewDirWS = GetWorldSpaceNormalizeViewDir(vertexInput.positionWS);                           
                output.normalWS = half3(normalInput.normalWS);
                float sign = input.tangentOS.w * float3(GetOddNegativeScale());
                output.tangentWS = half4(normalInput.tangentWS.xyz,sign);

                return output;               
            }

            half4 frag(Varyings input) : SV_Target
            {
                clip(1.0 - _AlphaClip);
                float3 normalWS = input.normalWS.xyz;
                return half4(NormalizeNormalPerPixel(normalWS),0.0);
            }
            ENDHLSL
            
        }
        Pass
        {
            Name"DrawNyxOutLine"
            Tags
            {
                "LightMode" = "UniversalForwardOnly"
            }
            Stencil
            {
                Ref 2
                Comp NotEqual
                Pass Keep
            }
            Cull Front
            ZWrite On

            HLSLPROGRAM
            #pragma shader_feature_local _OUTLINE_PASS
            #pragma shader_feature_local _NyxFire

            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fog
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            float GetCameraFOV()
            {
                float t = unity_CameraProjection._m11;
                float Rad2Deg = 180 / 3.1415;
                float fov = atan(1.0f / t) * 2.0 * Rad2Deg;
                return fov;
            }

            float ApplyOutlineDistanceFadeOut(float inputMulFix)
            {
                return saturate(inputMulFix);
            }

            float GetOutlineCameraFovAndDistanceFixMultiplier(float positionVS_Z)
            {
                float cameraMulFix;
                if(unity_OrthoParams.w == 0)
                {
                          
                    cameraMulFix = abs(positionVS_Z);

                    cameraMulFix = ApplyOutlineDistanceFadeOut(cameraMulFix);

                    cameraMulFix *= GetCameraFOV();       
                }
                else
                {
                    float orthoSize = abs(unity_OrthoParams.y);
                    orthoSize = ApplyOutlineDistanceFadeOut(orthoSize);
                    cameraMulFix = orthoSize * 50; // 50 is a magic number to match perspective camera's outline width
                }

                return cameraMulFix * 0.00005; // mul a const to make return result = default normal expand amount WS
            }
            

            float4 GetNewClipPosWithZOffset(float4 originalPositionCS, float viewSpaceZOffsetAmount)
            {
                if(unity_OrthoParams.w == 0)
                {
                   
                    float2 ProjM_ZRow_ZW = UNITY_MATRIX_P[2].zw;
                    float modifiedPositionVS_Z = -originalPositionCS.w + -viewSpaceZOffsetAmount; // push imaginary vertex
                    float modifiedPositionCS_Z = modifiedPositionVS_Z * ProjM_ZRow_ZW[0] + ProjM_ZRow_ZW[1];
                    originalPositionCS.z = modifiedPositionCS_Z * originalPositionCS.w / (-modifiedPositionVS_Z); // overwrite positionCS.z
                    return originalPositionCS;    
                }
                else
                {
                    originalPositionCS.z += -viewSpaceZOffsetAmount / _ProjectionParams.z; // push imaginary vertex and overwrite positionCS.z
                    return originalPositionCS;
                }
            }

            struct Attributes
            {
                float4 positionOS   : POSITION;
                float4 tangentOS    : TANGENT;
                float3 normalOS     : NORMAL;
                float2 texcoord     :TEXCOORD0;
                float2 texcoord1    :TEXCOORD1;
                float2 uv: TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS  : SV_POSITION;
                float fogFactor    : TEXCOORD1;
                float2 uv: TEXCOORD0;
                float3 positionWS : TEXCOORD2;
            };

            
            
            CBUFFER_START(UnityPerMaterial)
            float _NyxOutLineWidth;
            float _NyxMaxOutlineZoffset;
            float4 _BaseMap_ST;
            float4 _NyxNoiseSpeed;
            float _NyxNoiseIntensity;
            float4 _NyxLineSpeed;
            CBUFFER_END

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);
            TEXTURE2D(_NyxNoise);
            SAMPLER(sampler_NyxNoise);
            TEXTURE2D(_NyxRamp);
            SAMPLER(sampler_NyxRamp);

            Varyings vert(Attributes input)
            {   

                VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normalInputs = GetVertexNormalInputs(input.normalOS,input.tangentOS);
                
                float3 positionWS = positionInputs.positionWS.xyz;

                #if _NyxFire
                {
                // 火焰扰动UV，建议用世界坐标Y和时间做流动
                float2 flameUV = float2(positionWS.xz * _NyxNoiseSpeed.xy + _Time.y * _NyxNoiseSpeed.zw);
                

                // FBM多层噪声
                float noise = 0;
                float amplitude = 0.5;
                float frequency = 1.0;
                for (int i = 0; i < 2; i++)
                {
                    noise += SAMPLE_TEXTURE2D_LOD(_NyxNoise, sampler_NyxNoise, flameUV * frequency, 0).b * amplitude;
                    amplitude *= 0.5;
                    frequency *= 2.0;
                }
                noise = noise / (0.5 + 0.25); // 归一化

                // 火焰扰动主要沿法线和Y轴（世界上方向）
                float3 upDir = float3(0, 1, 0);
                float3 flameOffset = (normalInputs.tangentWS * 0.5 + upDir * 0.5) * (noise - 0.5) * _NyxNoiseIntensity * 0.5;

                positionWS += flameOffset;

                // 2. 叠加描边外扩（沿切线方向）
                float width = _NyxOutLineWidth;
                width *= GetOutlineCameraFovAndDistanceFixMultiplier(positionInputs.positionVS.z);
                positionWS += normalInputs.tangentWS * width;
            }
            #endif

                Varyings output = (Varyings)0;
                output.positionWS = positionWS;

                output.positionCS = GetNewClipPosWithZOffset(TransformWorldToHClip(positionWS),_NyxMaxOutlineZoffset);
                output.uv = input.texcoord;
                output.fogFactor = ComputeFogFactor(positionInputs.positionCS.z);


                return output;
            }

            float4 frag(Varyings input) : SV_Target
            {

                float3 positionWS = input.positionWS;
                float2 c = (positionWS.xyz - TransformObjectToWorld(float3(0, 0, 0)).xyz).xy;
                    c *= _NyxLineSpeed.xy; 
                    float2 NyxpannerUV = c + _Time.y * _NyxLineSpeed.zw; 
                    float NyxLightMap = SAMPLE_TEXTURE2D(_NyxNoise, sampler_NyxNoise, NyxpannerUV).r;
                    NyxLightMap = saturate(NyxLightMap); // 保证在0~1
                    float2 outlineColorUV = float2(NyxLightMap, frac(_Time.y * 0.1));
                    float3 outlineColor = SAMPLE_TEXTURE2D(_NyxRamp, sampler_NyxRamp, outlineColorUV).rgb;
                    outlineColor *= 3;
                    //float3 outlineColor = 1;
                    float4 color = float4(0,0,0,0);
                #if _NyxFire
                {
                color = float4(outlineColor,1);
                }
                #endif
                
                //color.rgb = MixFog(color.rgb, input.fogFactor);

                return color;
            }
            ENDHLSL
        }
        Pass
        {
            Name"DrawOutLine"
            Tags
            {
                "LightMode" = "SRPDefaultUnlit"
            }
            Cull Front
            ZWrite On

            HLSLPROGRAM
            #pragma shader_feature_local _OUTLINE_PASS

            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fog
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            float GetCameraFOV()
            {
                float t = unity_CameraProjection._m11;
                float Rad2Deg = 180 / 3.1415;
                float fov = atan(1.0f / t) * 2.0 * Rad2Deg;
                return fov;
            }

            float ApplyOutlineDistanceFadeOut(float inputMulFix)
            {
                return saturate(inputMulFix);
            }

            float GetOutlineCameraFovAndDistanceFixMultiplier(float positionVS_Z)
            {
                float cameraMulFix;
                if(unity_OrthoParams.w == 0)
                {
                          
                    cameraMulFix = abs(positionVS_Z);

                    cameraMulFix = ApplyOutlineDistanceFadeOut(cameraMulFix);

                    cameraMulFix *= GetCameraFOV();       
                }
                else
                {
                    float orthoSize = abs(unity_OrthoParams.y);
                    orthoSize = ApplyOutlineDistanceFadeOut(orthoSize);
                    cameraMulFix = orthoSize * 50; // 50 is a magic number to match perspective camera's outline width
                }

                return cameraMulFix * 0.00005; // mul a const to make return result = default normal expand amount WS
            }
            

            float4 GetNewClipPosWithZOffset(float4 originalPositionCS, float viewSpaceZOffsetAmount)
            {
                if(unity_OrthoParams.w == 0)
                {
                   
                    float2 ProjM_ZRow_ZW = UNITY_MATRIX_P[2].zw;
                    float modifiedPositionVS_Z = -originalPositionCS.w + -viewSpaceZOffsetAmount; // push imaginary vertex
                    float modifiedPositionCS_Z = modifiedPositionVS_Z * ProjM_ZRow_ZW[0] + ProjM_ZRow_ZW[1];
                    originalPositionCS.z = modifiedPositionCS_Z * originalPositionCS.w / (-modifiedPositionVS_Z); // overwrite positionCS.z
                    return originalPositionCS;    
                }
                else
                {
                    originalPositionCS.z += -viewSpaceZOffsetAmount / _ProjectionParams.z; // push imaginary vertex and overwrite positionCS.z
                    return originalPositionCS;
                }
            }

            struct Attributes
            {
                float4 positionOS   : POSITION;
                float4 tangentOS    : TANGENT;
                float3 normalOS     : NORMAL;
                float2 texcoord     :TEXCOORD0;
                float2 texcoord1    :TEXCOORD1;
                float2 uv: TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS  : SV_POSITION;
                float fogFactor    : TEXCOORD1;
                float2 uv: TEXCOORD0;
            };

            
            
            CBUFFER_START(UnityPerMaterial)
            float _OutLineWidth;
            float _MaxOutlineZoffset;
            //float _MaterialIDUSE; // 添加材质ID变量
            float4 _OutlineColor0;
            float4 _OutlineColor1;
            float4 _OutlineColor2;
            float4 _OutlineColor3;
            float4 _OutlineColor4;
            //float4 _Color;
            float4 _BaseMap_ST;
            CBUFFER_END

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);
            TEXTURE2D(_ILMTex);
            SAMPLER(sampler_ILMTex);

            Varyings vert(Attributes input)
            {   

                VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normalInputs = GetVertexNormalInputs(input.normalOS,input.tangentOS);

                float width = _OutLineWidth;
                width *= GetOutlineCameraFovAndDistanceFixMultiplier(positionInputs.positionVS.z);

                float3 positionWS = positionInputs.positionWS.xyz;
                positionWS += normalInputs.tangentWS * width;

                Varyings output = (Varyings)0;
                output.positionCS = GetNewClipPosWithZOffset(TransformWorldToHClip(positionWS),_MaxOutlineZoffset);
                output.uv = input.texcoord;
                output.fogFactor = ComputeFogFactor(positionInputs.positionCS.z);


                return output;
            }

            float4 frag(Varyings input) : SV_Target
            {
                float4 ilm = SAMPLE_TEXTURE2D(_ILMTex, sampler_ILMTex, input.uv);

                float MatEnum0 = 0.0;
                float MatEnum1 = 0.3;
                float MatEnum2 = 0.5;
                float MatEnum3 = 0.7;
                float MatEnum4 = 1.0;

                float4 Color = lerp(_OutlineColor4, _OutlineColor3, step(ilm.a, (MatEnum3 + MatEnum4) / 2));
                Color = lerp(Color, _OutlineColor2, step(ilm.a, (MatEnum2 + MatEnum3) / 2));
                Color = lerp(Color, _OutlineColor1, step(ilm.a, (MatEnum1 + MatEnum2) / 2));
                Color = lerp(Color, _OutlineColor0, step(ilm.a, (MatEnum0 + MatEnum1) / 2));

                float3 outlineColor = Color.rgb;
            

                float4 color = float4(outlineColor,1);
                color.rgb = MixFog(color.rgb, input.fogFactor);

                return color;
            }
            ENDHLSL
        }
       // 阴影投射Pass
        UsePass "Universal Render Pipeline/Lit/ShadowCaster" 
    } 
    FallBack "Diffuse"
}
