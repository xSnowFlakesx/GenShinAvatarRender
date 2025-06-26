Shader "Unlit/Genshin_Face_Render"
{
    Properties //着色器的输入 
    {
        [Header(BaseColor)]
        _BaseMap ("Base Color", 2D) = "white" {}
        _DiffuseColor ("Diffuse Color", Color) = (1,1,1,1)
        _AmbientColor ("Ambient Color", Color) = (0.5,0.5,0.5,1)
        _ShadowColor ("Shadow Color", Color) = (0.7,0.7,0.7,1)
        _AmbientIntensity ("Ambient Intensity", Range(0, 1)) = 0.5
        _SSSColor ("SSS Color", Color) = (1,1,1,1)
       
        
        [Header(SDF)]
        _SDFTex ("SDF Tex", 2D) = "white" {}
        _ShadowMaskTex ("Shadow Mask", 2D) = "white" {}

        [Header(ColorMapping)]
        _ToonTex("Toon Tex", 2D) = "white" {}
        _BaseTexFac("BaseTexFac", Range(0, 1)) = 1
        _ToonTexFac("ToonTexFac", Range(0, 1)) = 1
        _SphereTexFac("SphereTexFac", Range(0, 1)) = 0
        _SphereMulAdd("SphereMul/Add", Range(0, 1)) = 0
        _SphereTex("Sphere Tex", 2D) = "white" {}


        [Header(RampColor)]
        _RampTex ("Ramp Texture", 2D) = "white" {}
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

        [Header(HeadVector)]
        [HideInInspector]_HeadCenter("Head Center", Vector) = (0,0,0,0)
        [HideInInspector]_HeadForward("Head Forward", Vector) = (0,0,1,0)
        [HideInInspector]_HeadRight("Head Right", Vector) = (1,0,0,0)

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
    }
    SubShader
    {
        LOD 100

        Pass
        {
        Name "ForwardBase" //渲染通道名称  

        Tags {
            "LightMode"="UniversalForward" //用于指明渲染模式
            "RenderType"="Opaque" //用于指明渲染类型
            "RenderPipeLine"="UniversalRenderPipeline" //用于指明使用URP来渲染
        }

        ZWrite On //开启深度写入
        Cull Off //关闭剔除
        Blend SrcAlpha OneMinusSrcAlpha //开启混合

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
            float3 _HeadCenter;
            float3 _HeadForward;
            float3 _HeadRight;
            float4 _FlowTillingSpeed;
            float _RimOffset;
            float4 _RimColor;
            float _RimIntensity;
            float _RimThreshold;
            float4 _SSSColor;
        CBUFFER_END

        TEXTURE2D(_BaseMap); //贴图采样  
        SAMPLER(sampler_BaseMap);
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
        TEXTURE2D(_SDFTex);
        SAMPLER(sampler_SDFTex);
        TEXTURE2D(_ShadowMaskTex);
        SAMPLER(sampler_ShadowMaskTex);
        TEXTURE2D(_FlowMask);
        SAMPLER(sampler_FlowMask);
        TEXTURE2D(_FlowMap);
        SAMPLER(sampler_FlowMap);
        TEXTURE2D(_FlowRamp);
        SAMPLER(sampler_FlowRamp);

        float max3(float a, float b, float c) //max3
{
    return max(max(a, b), c);
}

        float AverageColor(float3 color) //AverageColor
            {
                return dot(color,float3(1.0,1.0,1.0))/3.0;
            }

            float3 NormalizeColorByAverage(float3 color) //NormalizeColorByAverage
            {
                float average = AverageColor(color);
                return color / max(average,1e-5);
            }

            float3 ScaleColorByMax(float3 color) //ScaleColorByMax
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
                half4 vertexColor: COLOR;
            };

            Varyings vert (Attributes input)//顶点着色器
            {
                Varyings output;

                VertexPositionInputs vertexInput = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionNDC = vertexInput.positionNDC;

                // 初始化 positionCS
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);

                // 初始化 uv
                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);

                // 初始化 vertexColor
                output.vertexColor = input.vertexColor;

                // 初始化其他字段
                output.positionWS = TransformObjectToWorld(input.positionOS.xyz).xyz;
                output.positionVS = TransformWorldToView(output.positionWS);
                output.normalWS = TransformObjectToWorldNormal(input.normalOS);
                output.tangentWS = TransformObjectToWorldDir(input.tangentOS.xyz);
                output.bitangentWS = cross(output.normalWS, output.tangentWS) * input.tangentOS.w;

                return output;
            }

            half4 frag (Varyings input) : SV_Target  // 片元着色器
            {

                //  shadow
                float4 shadowCoord = TransformWorldToShadowCoord(input.positionWS);
                Light mainLight = GetMainLight(shadowCoord);

                // light
                float3 lightColor = mainLight.color.rgb;
                lightColor = ScaleColorByMax(lightColor);

                
                //向量计算
                float3 N = normalize(input.normalWS);
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

                //  MatCapUV
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

                //RampV坐标
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

                
                //HeadDirection
                // 根据头部三个定位点构建头部的局部坐标系
                float3 headForward = normalize(_HeadForward - _HeadCenter);
                float3 headRight = normalize(_HeadRight - _HeadCenter);
                float3 headUp = normalize(cross(headForward, headRight));

                float3 LpU = dot(L, headUp) / pow(length(headUp), 2) * headUp; //光照方向在头部上方向量的投影
                float3 LpHeadHorizon = L - LpU; //光照方向在头部水平面的投影分量

                float Value = acos(dot(normalize(LpHeadHorizon),normalize(headRight))) / PI; //计算光照方向在头部水平面的投影分量与头部右侧向量的夹角，并归一化到[0,1]

                float exposeRight = step(Value, 0.5);

                float ValueR = pow(1 - Value * 2, 3);//三次方衰减曲线
                float ValueL = pow(Value * 2 - 1, 3);//三次方递增曲线
                float mixValue = lerp(ValueL, ValueR, exposeRight);

                // 镜像采样SDF贴图（左右对称处理）
                float sdfRembrandLeft = SAMPLE_TEXTURE2D(_SDFTex,sampler_SDFTex, float2(1 - input.uv.x, input.uv.y)).a;
                float sdfRembrandRight = SAMPLE_TEXTURE2D(_SDFTex,sampler_SDFTex, input.uv).a;
                float mixSDF = lerp(sdfRembrandRight, sdfRembrandLeft, exposeRight);

                float width = 0.007; 
                float SDF = step(mixValue - width, mixSDF);
                float SDF2 = step(mixValue + width, mixSDF);
                SDF = lerp(SDF,SDF2,mixSDF);

                // 排除背面光照区域的阴影（只保留正面投影）
                SDF = lerp(0, SDF, step(0, dot(normalize(LpHeadHorizon), normalize(headForward))));
                SDF2 = lerp(0, SDF, step(0, dot(normalize(LpHeadHorizon), normalize(headForward))));//多采样一次做伪SSS效果

                float4 shadowTex = SAMPLE_TEXTURE2D(_ShadowMaskTex, sampler_ShadowMaskTex, input.uv);

                SDF *= shadowTex.g; 
                SDF = lerp(SDF, 1, shadowTex.a); 
                SDF *= shadowAttenuation; 

                SDF2 *= shadowTex.g; 
                SDF2 = lerp(SDF, 1, shadowTex.a);
                //SDF2 *= shadowAttenuation;

                // 白天渐变控制
                float RampV = _RampMapRow4/10 - 0.05; 
                float RampU = clamp(SDF, 0.005,0.995); 

                float RampV2 = _RampMapRow1/10 - 0.05; 
                float RampU2 = clamp(SDF2, 0.005,0.995);
                float2 RampDayUV = float2(RampU, 1 - RampV);
                float2 RampNightUV = float2(RampU, 1 - (RampV + 0.5));

                float2 RampDayUV2 = float2(RampU2, 1 - RampV2);
                float2 RampNightUV2 = float2(RampU2, 1 - (RampV2 + 0.5));

                float Day = (L.y + 1) / 2;// 根据光源高度判断昼夜（L.y=1为白天，-1为夜晚）
                float3 RampColor = lerp(SAMPLE_TEXTURE2D(_RampTex,sampler_RampTex,RampNightUV).rgb,SAMPLE_TEXTURE2D(_RampTex,sampler_RampTex,RampDayUV).rgb,Day); 

                float3 RampColor2 = lerp(SAMPLE_TEXTURE2D(_RampTex,sampler_RampTex,RampNightUV2).rgb,SAMPLE_TEXTURE2D(_RampTex,sampler_RampTex,RampDayUV2).rgb,Day);

                RampColor2 *= _SSSColor.rgb * 5;

                RampColor = lerp(RampColor, RampColor2, SDF);

                float3 shadowColor = baseColor * RampColor * _ShadowColor.rgb; 

                float3 diffuse = lerp(shadowColor,baseColor,SDF); 


                //Flow Light
                float FlowMaskFace = 0;
                float FlowMap = 0;
                float3 FlowColor = 0;
                #if _FLOWLight
                {        
                    FlowMaskFace = SAMPLE_TEXTURE2D(_FlowMask, sampler_FlowMask, input.uv).b;

                    float2 a = ((input.positionWS).xyz - TransformObjectToWorld(float3(0, 0, 0)).xyz).xy;//计算物体原点世界坐标，生成基于世界坐标的平面位移向量（忽略Y轴高度变化）
                    a *= _FlowTillingSpeed.xy; // 通过xy平面坐标差生成基础UV，实现基于物体位置的流动效果

                    float2 pannerUV = a + _Time.y * _FlowTillingSpeed.zw; 

                    FlowMap = SAMPLE_TEXTURE2D(_FlowMap, sampler_FlowMap, pannerUV).g;
                    FlowMap = saturate(FlowMap); 

                    float2 FlowRampUV = float2(FlowMap, 0.5);
                    FlowColor = SAMPLE_TEXTURE2D(_FlowRamp, sampler_FlowRamp, FlowRampUV).rgb;

                    FlowColor *= FlowMaskFace;
                    FlowColor = ScaleColorByMax(FlowColor);
                }
                #endif

                //Screen Rim
                float2 screenUV = input.positionNDC.xy / input.positionNDC.w;
                // 采样当前像素的原始深度并转换为线性眼空间深度
                float rawDepth = SampleSceneDepth(screenUV);
                float linearDepth = LinearEyeDepth(rawDepth, _ZBufferParams);

                float2 screenOffset = float2(lerp(-1, 1, step(0, normalVS.x)) * _RimOffset / _ScreenParams.x / max(1,pow(linearDepth, 2)), 0);

                float offsetDepth = SampleSceneDepth(screenUV + screenOffset);
                float offsetLinearDepth = LinearEyeDepth(offsetDepth, _ZBufferParams);

                float Rim = saturate(offsetLinearDepth - linearDepth); 
                Rim = step(_RimThreshold, Rim) * _RimIntensity; 
                
                
                float fresnelPower = 6; 
                float fresnelClamp = 0.8; 
                float fresnel = 1 - saturate(NoV); 
                fresnel = pow(fresnel, fresnelPower); 
                fresnel = fresnel * fresnelClamp + (1 - fresnelClamp);

                float3 albedo = diffuse;
                albedo =  1 - (1 - Rim * fresnel) * (1 - albedo); 
                albedo +=  FlowColor;
                albedo *= lightColor;


                //环境光
                float3 ambient = SampleSH(normalize(lerp(N, float3(1, 1, 1), 0.7)));

                // 只在阴影区叠加环境光
                albedo = lerp(albedo, albedo + ambient * _AmbientIntensity, 1 - SDF);

                //多光源
                int additionalLightsCount = GetAdditionalLightsCount();
                for (int i = 0; i < additionalLightsCount; ++i)
                {
                    Light light = GetAdditionalLight(i, input.positionWS);
                    float3 L = normalize(light.direction);
                    //float NoL = saturate(dot(N, L));
                    float shadowAtten = light.shadowAttenuation;
                    float3 addDiffuse = SDF * light.color.rgb * shadowAtten;
                    albedo += addDiffuse * baseColor;
                }

                half4 col = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv);
                return float4(albedo,1);
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

            #pragma multi_compile_instancing
            #pragma multi_compile_DOTS_INSTANCING_ON

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

            float _AlphaClip;

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                return OUT;
            }

            float4 fragDepth(Varyings IN) : SV_Target
            {
                clip(1.0 - _AlphaClip);
                return float4(0,0,0,0);
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

            #pragma multi_compile_instancing
            #pragma multi_compile_DOTS_INSTANCING_ON

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

            float _AlphaClip;

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
            Name"DrawOutLine"
            Tags
            {
                "LightMode" = "UniversalForwardOnly"
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
