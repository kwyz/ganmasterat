<template>
<v-app>
    <v-app-bar app color="dark-grey" dark  class="text-center">
        <h2 class="text-center text-uppercase col-12">
            Rețea neurală competitiv-generativă
        </h2>
    </v-app-bar>
    <v-content>
      <v-container>
        <v-row justify="center">
          <v-col cols="12">
          </v-col>
        </v-row>
      </v-container>
        <v-container>
            <v-row justify="center"> 
                <Canvas/>
                <Controll/>
            </v-row>
      </v-container>
    </v-content>
  </v-app>
</template>

<script>

export default {
  name: 'App',
  components: {
    Canvas: () => import("@/components/Canvas/Index.vue"),
    Controll: () => import("@/components/Controll/Index.vue"),
  },

  data: () => ({
    //
  }),
};
</script>
